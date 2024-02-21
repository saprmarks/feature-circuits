import argparse
import os
import pickle
import random
import torch as t
from graph_utils import WeightedDAG, deduce_edge_weights
from attribution import patching_effect, get_grad
from tensordict import TensorDict

from nnsight import LanguageModel
from tqdm import tqdm, trange
from copy import deepcopy
from collections import defaultdict
from loading_utils import (
    load_examples, load_submodule_and_dictionary, submodule_name_to_type_layer, DictionaryCfg
)
from acdc import patching_on_y, patching_on_downstream_feature
from ablation_utils import run_with_ablated_features

class CircuitNode:
    def __init__(self, name, submodule, dictionary, feat_idx):
        self.name = name
        self.submodule = submodule
        self.dictionary = dictionary
        self.feat_idx = feat_idx
    
    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return self.name
    
    def __eq__(self, other):
        return self.name == other.name

def get_circuit(
        clean, 
        patch,
        model,
        submodules,
        submodule_names,
        dictionaries,
        metric_fn,
        node_threshold=0.1,
        edge_threshold=0.01,
        method='separate',
):
    
    # first, figure out which features get to go in the circuit and the gradient of y wrt these feeatures
    effects, deltas, grads, total_effect = patching_effect(clean, patch, model, submodules, dictionaries, metric_fn, method=method)
    feat_idxs = {}
    grads_to_y = {}
    for submodule, effect in effects.items():
        feat_idxs[submodule] = t.nonzero(effect > node_threshold)
        grads_to_y[submodule] = TensorDict({idx : grads[submodule][tuple(idx)] for idx in feat_idxs[submodule]})
        
    # construct DAG
    dag = WeightedDAG()
    grads = {} # stores grads between any two pairs of nodes

    y = CircuitNode("y", None, None, None)
    dag.add_node(y, total_effect)

    nodes_by_component = {}
    for submodule, idxs in reversed(feat_idxs.items()): # assuming that submodules are properly ordered)
        # get nodes for this component
        nodes = []
        for idx in idxs:
            n = CircuitNode(f'{submodule_names[submodule]}/{idx}', submodule, dictionaries[submodule], idx)
            nodes.append(n)
        nodes_by_component[submodule] = nodes

        # add nodes to the DAG
        old_nodes = dag.nodes.copy()
        for n in nodes:
            dag.add_node(n, weight=deltas[submodule][tuple(n.feat_idx)])
            grads[(y, n)] = grads_to_y[submodule][n.feat_idx]
            for n_old in old_nodes:
                dag.add_edge(n, n_old)

    # compute the gradients of the upstream features wrt the downstream features
    for downstream_idx, downstream_submod in reversed(list(enumerate(submodules))):
        for upstream_submod in submodules[:downstream_idx]:
            upstream_grads = get_grad(
                clean,
                patch,
                model,
                upstream_submod,
                dictionaries[upstream_submod],
                downstream_submod,
                dictionaries[downstream_submod],
                feat_idxs[downstream_submod],
            )
            for downstream_node in nodes_by_component[downstream_submod]:
                for upstream_node in nodes_by_component[upstream_submod]:
                    grads[(downstream_node, upstream_node)] = upstream_grads[downstream_node.feat_idx][tuple(upstream_node.feat_idx)]



    # compute the edge weights
    dag = deduce_edge_weights(dag, grads)

    # reassign weights to represent attribution scores
    for n1, n2 in dag.edges:
        if n2 == y:
            dag.add_edge(n1, n2, weight=dag.node_weight(n1) * dag.edge_weight((n1, n2)))
        else:
            dag.add_edge(n1, n2, weight=dag.node_weight(n1) * dag.edge_weight((n1, n2)) * grads_to_y[n2.submodule][n2.feat_idx])
    for n in dag.nodes:
        if n == y: continue
        dag.add_node(n, weight=dag.node_weight(n) * grads_to_y[n.submodule][n.feat_idx])

    # filter edges
    for n1, n2 in dag.edges:
        if dag.edge_weight((n1, n2)) < edge_threshold:
            dag.remove_edge((n1, n2))

    return dag

def slice_dag(dag, pos, dim):
    """
    Given a DAG whose nodes are CircuitNodes, returns a new DAG consisting only of those nodes whose feat_idx[dim] == pos
    """
    new_dag = WeightedDAG()
    for n in dag.nodes:
        if n.feat_idx is None or n.feat_idx[dim] == pos:
            new_dag.add_node(n, weight=dag.node_weight(n))
    for n1, n2 in dag.edges:
        if n1 in new_dag.nodes and n2 in new_dag.nodes:
            new_dag.add_edge(n1, n2, weight=dag.edge_weight((n1, n2)))
    return new_dag

def _remove_coord(x : t.Tensor, pos : int):
    """
    Given a tensor x, returns a new tensor with the pos-th coordinate removed
    """
    return t.cat([x[:pos], x[pos+1:]], dim=0)

def _sum_with_crosses(dag, dim):
    new_dag = dag.copy()
    topo_order = dag.topological_sort()
    for n in topo_order:
        if n.name == 'y':
            continue
        feat_idx = _remove_coord(n.feat_idx, dim)
        name = f"{'/'.join(n.name.split('/')[:-1])}/{feat_idx}"
        n_summed = CircuitNode(name, n.submodule, n.dictionary, feat_idx)
        if new_dag.has_node(n_summed):
            new_dag.add_node(n_summed, weight=new_dag.node_weight(n_summed) + new_dag.node_weight(n))
        else:
            new_dag.add_node(n_summed, weight=new_dag.node_weight(n))
        for m in new_dag.get_children(n):
            if new_dag.has_edge((n_summed, m)):
                new_dag.add_edge(n_summed, m, weight=new_dag.edge_weight((n_summed, m)) + new_dag.edge_weight((n, m)))
            else:
                new_dag.add_edge(n_summed, m, weight=new_dag.edge_weight((n, m)))
        for m in new_dag.get_parents(n):
            if new_dag.has_edge((m, n_summed)):
                new_dag.add_edge(m, n_summed, weight=new_dag.edge_weight((m, n_summed)) + new_dag.edge_weight((m, n)))
            else:
                new_dag.add_edge(m, n_summed, weight=new_dag.edge_weight((m, n)))
        new_dag.remove_node(n)
    
    return new_dag

def _sum_without_crosses(dag, dim):
    new_dag = WeightedDAG()
    for n in dag.nodes:
        if n.name == 'y':
            new_dag.add_node(n, weight=dag.node_weight(n).sum())
        else:
            feat_idx = _remove_coord(n.feat_idx, dim)
            name = f"{'/'.join(n.name.split('/')[:-1])}/{feat_idx}"
            n_summed = CircuitNode(name, n.submodule, n.dictionary, feat_idx)
            if new_dag.has_node(n_summed):
                new_dag.add_node(n_summed, weight=new_dag.node_weight(n_summed) + dag.node_weight(n))
            else:
                new_dag.add_node(n_summed, weight=dag.node_weight(n))
    for e in dag.edges:
        n1, n2 = e
        if n2.name == 'y':
            n1_summed_name = f"{'/'.join(n1.name.split('/')[:-1])}/{_remove_coord(n1.feat_idx, dim)}"
            n1_summed = CircuitNode(n1_summed_name, n1.submodule, n1.dictionary, _remove_coord(n1.feat_idx, dim))
            n2_summed = n2
        elif n1.feat_idx[dim] != n2.feat_idx[dim]:
            raise ValueError(f"crosses was set to False, but some edge crosses positions in dimension {dim}")
        else:
            n1_summed_name = f"{'/'.join(n1.name.split('/')[:-1])}/{_remove_coord(n1.feat_idx, dim)}"
            n2_summed_name = f"{'/'.join(n2.name.split('/')[:-1])}/{_remove_coord(n2.feat_idx, dim)}"
            n1_summed = CircuitNode(n1_summed_name, n1.submodule, n1.dictionary, _remove_coord(n1.feat_idx, dim))
            n2_summed = CircuitNode(n2_summed_name, n2.submodule, n2.dictionary, _remove_coord(n2.feat_idx, dim))
        if new_dag.has_edge((n1_summed, n2_summed)):
            new_dag.add_edge(n1_summed, n2_summed, weight=new_dag.edge_weight((n1_summed, n2_summed)) + dag.edge_weight(e))
        else:
            new_dag.add_edge(n1_summed, n2_summed, weight=dag.edge_weight(e))
    return new_dag

def sum_dag(dag, dim, crosses=True):
    """
    Given a DAG whose nodes are CircuitNodes, produce a new DAG whose:
    - nodes are the sum of the nodes in the original DAG, where the sum is taken over the dim-th index of the feat_idx
    - edges (n1, n2) are the sum of the edges whose endpoints were summed into n1 and n2
    If crosses is True, then the new DAG will have edges between nodes that are not in the same position in the dim-th index of the feat_idx
    """
    if crosses:
        return _sum_with_crosses(dag, dim)
    else:
        return _sum_without_crosses(dag, dim)
    
def _mean_with_crosses(dag, dim):
    new_dag = dag.copy()
    topo_order = dag.topological_sort()
    counts = defaultdict(int)
    for n in topo_order:
        if n.feat_idx is None:
            continue
        feat_idx = _remove_coord(n.feat_idx, dim)
        name = f"{'/'.join(n.name.split('/')[:-1])}/{feat_idx}"
        n_summed = CircuitNode(name, n.submodule, n.dictionary, feat_idx)

        if new_dag.has_node(n_summed):
            new_dag.add_node(n_summed, weight=new_dag.node_weight(n_summed) + new_dag.node_weight(n))
        else:
            new_dag.add_node(n_summed, weight=new_dag.node_weight(n))
        counts[n_summed] += 1

        for m in new_dag.get_children(n):
            if new_dag.has_edge((n_summed, m)):
                new_dag.add_edge(n_summed, m, weight=new_dag.edge_weight((n_summed, m)) + new_dag.edge_weight((n, m)))
            else:
                new_dag.add_edge(n_summed, m, weight=new_dag.edge_weight((n, m)))
            counts[(n_summed, m)] += 1
        for m in new_dag.get_parents(n):
            if new_dag.has_edge((m, n_summed)):
                new_dag.add_edge(m, n_summed, weight=new_dag.edge_weight((m, n_summed)) + new_dag.edge_weight((m, n)))
            else:
                new_dag.add_edge(m, n_summed, weight=new_dag.edge_weight((m, n)))
            counts[(m, n_summed)] += 1
        new_dag.remove_node(n)
    
    for n in new_dag.nodes:
        new_dag.add_node(n, weight=new_dag.node_weight(n) / counts[n])
    for e in new_dag.edges:
        new_dag.add_edge(e[0], e[1], weight=new_dag.edge_weight(e) / counts[e])

    return new_dag

def mean_dag(dag, dim, crosses=True):
    """
    Given a DAG whose nodes are CircuitNodes, produce a new DAG whose:
    - nodes are the mean of the nodes in the original DAG, where the mean is taken over the dim-th index of the feat_idx
    - edges (n1, n2) are the mean of the edges whose endpoints were summed into n1 and n2
    If crosses is True, then the new DAG will have edges between nodes that are not in the same position in the dim-th index of the feat_idx
    """
    if crosses:
        return _mean_with_crosses(dag, dim)
    else:
        dim_size = max([n.feat_idx[dim] for n in dag.nodes if n.feat_idx is not None]) + 1
        new_dag = _sum_without_crosses(dag, dim)
        for n in new_dag.nodes:
            new_dag.add_node(n, weight=new_dag.node_weight(n) / dim_size)
        for e in new_dag.edges:
            new_dag.add_edge(e[0], e[1], weight=new_dag.edge_weight(e) / dim_size)
        return new_dag



                    def evaluate_faithfulness(self, eval_dataset=None, patch_type='zero'):
        """
        Evaluate performance of circuit compared to full model.
        `patch_type` can be one of the following:
        - "zero": sets activation to zero
        - "mean": sets activation to its mean over many Pile contexts (loads from .pkl)
        - "random": sets activation to what it would've been given a single Pile
                    context (computed in-function)
        """
        if not eval_dataset:    # evaluate on train dataset
            eval_dataset = self.dataset
        feature_list = self.get_feature_list()

        if patch_type == "zero":
            patch_vector = t.zeros(self.dict_cfg.size)
        elif patch_type == "mean":
            raise NotImplementedError()
        elif patch_type == "random":
            raise NotImplementedError()
        
        # Pre-load all submodules and dictionaries
        num_layers = self.model.config.num_hidden_layers
        submodule_to_autoencoder = {}
        for layer in range(num_layers):
            for submodule_name in self.submodules_generic:
                submodule_name = submodule_name.format(str(layer))
                submodule, dictionary = load_submodule_and_dictionary(self.model, submodule_name, self.dict_cfg)
                submodule_to_autoencoder[submodule] = dictionary
        
        mean_faithfulness = 0.
        num_examples = len(eval_dataset)
        faithfulness_by_example = {}
        for example in tqdm(eval_dataset, desc="Evaluating faithfulness", total=len(eval_dataset)):
            with self.model.invoke(example["clean_prefix"]) as invoker:
                pass
            model_logit_diff = invoker.output.logits[:, -1, example["clean_answer"]] - \
                                invoker.output.logits[:, -1, example["patch_answer"]]

            circuit_out = run_with_ablated_features(self.model, example["clean_prefix"], self.dict_cfg.dir, self.dict_cfg.size,
                                                    feature_list, submodule_to_autoencoder, 
                                                    patch_vector=patch_vector, inverse=True)["model"]
            circuit_logit_diff = circuit_out.logits[:, -1, example["clean_answer"]] - \
                                    circuit_out.logits[:, -1, example["patch_answer"]]
            faithfulness = circuit_logit_diff / model_logit_diff

            # print(example["clean_prefix"], example["clean_answer"])
            # example_sent = self.model.tokenizer.decode(example["clean_prefix"][0]) + " " + self.model.tokenizer.decode(example["clean_answer"])
            # faithfulness_by_example[example_sent] = faithfulness
            mean_faithfulness += faithfulness
        
        # sorted_faithfulness = {k: v for k, v in sorted(faithfulness_by_example.items(), key=lambda x: x[1])}
        # for example in sorted_faithfulness:
        #     print(f"{example}: {sorted_faithfulness[example]}")

        mean_faithfulness /= num_examples
        return mean_faithfulness.item()


    def evaluate_completeness(self, eval_dataset=None, patch_type='zero', K_size=10, sample_size=5):
        """
        Evaluate whether we've .
        `patch_type` can be one of the following:
        - "zero": sets activation to zero
        - "mean": sets activation to its mean over many Pile contexts (loads from .pkl)
        - "random": sets activation to what it would've been given a single Pile
                    context (computed in-function)
        """
        if not eval_dataset:    # evaluate on train dataset
            eval_dataset = self.dataset
        circuit_feature_set = set(self.get_feature_list())

        if patch_type == "zero":
            patch_vector = t.zeros(self.dict_cfg.size)
        elif patch_type == "mean":
            raise NotImplementedError()
        elif patch_type == "random":
            raise NotImplementedError()

        # Pre-load all submodules and dictionaries
        num_layers = self.model.config.num_hidden_layers
        submodule_to_autoencoder = {}
        for layer in range(num_layers):
            for submodule_name in self.submodules_generic:
                submodule_name = submodule_name.format(str(layer))
                submodule, dictionary = load_submodule_and_dictionary(self.model, submodule_name, self.dict_cfg)
                submodule_to_autoencoder[submodule] = dictionary
        
        mean_percent_recovered = 0
        completeness_points = []
        mean_incompleteness = 0.
        total = 0
        K = set()
        num_examples = len(eval_dataset)
        curr_parent = self.root
        next_parents = []      # queue
        for _ in tqdm(range(K_size), desc="Building K", total=K_size):
            curr_node = None
            while True:     # do-while loop
                max_effect = float("-inf")
                for child in curr_parent.children:
                    next_parents.append(child)
                    if self._normalize_name(child.name) in K:
                        continue
                    if child.effect_on_parents[curr_parent] > max_effect:
                        max_effect = child.effect_on_parents[curr_parent]
                        curr_node = child
                if curr_node is None:
                    curr_parent = next_parents.pop(0)
                if not (curr_node is None and len(next_parents) != 0):
                    break
            if curr_node is None:
                print(f"No more nodes to add. Exiting loop w/ |K|={len(K)}")
            else:
                K.add(self._normalize_name(curr_node.name))
            
            # subsample_size = min(sample_size, len(circuit_features_no_K))
            # feature_subsample = random.choices(list(circuit_features_no_K), k=subsample_size)
            # max_incompleteness_feature = None
            # final_circuit_no_K_diff = None
            # max_incompleteness = float("-inf")
            # greedily locate feature in subsample that maximizes incompleteness score
            # for feature in feature_subsample:
            #     circuit_features_no_K_v = deepcopy(circuit_features_no_K)
            #     circuit_features_no_K_v.remove(feature)
            #     mean_change_in_diff = 0.
            #     for example in eval_dataset:
            #         circuit_no_K_out = run_with_ablated_features(self.model, example["clean_prefix"],
            #                                                         self.dict_cfg.dir, self.dict_cfg.size,
            #                                                         list(circuit_features_no_K), submodule_to_autoencoder,
            #                                                         patch_vector=patch_vector, inverse=True)["model"]
            #         circuit_no_K_v_out = run_with_ablated_features(self.model, example["clean_prefix"],
            #                                                         self.dict_cfg.dir, self.dict_cfg.size,
            #                                                         list(circuit_features_no_K_v), submodule_to_autoencoder, 
            #                                                         patch_vector=patch_vector, inverse=True)["model"]
            #         circuit_no_K_diff = circuit_no_K_out.logits[:, -1, example["clean_answer"]] - \
            #                             circuit_no_K_out.logits[:, -1, example["patch_answer"]]
            #         circuit_no_K_v_diff = circuit_no_K_v_out.logits[:, -1, example["clean_answer"]] - \
            #                                 circuit_no_K_v_out.logits[:, -1, example["patch_answer"]]
            #         mean_change_in_diff += abs(circuit_no_K_v_diff - circuit_no_K_diff)
            #     mean_change_in_diff /= num_examples
            #     if mean_change_in_diff > max_incompleteness:
            #         max_incompleteness = mean_change_in_diff
            #         max_incompleteness_feature = feature
            # K.add(max_incompleteness_feature)

        # compute incompleteness
        model_no_K_diff = 0.
        circuit_features_no_K = circuit_feature_set.difference(K)
        completeness = 0.
        for example in tqdm(eval_dataset, desc="Evaluating completeness", total=len(eval_dataset)):
            model_no_K_out = run_with_ablated_features(self.model, example["clean_prefix"],
                                        self.dict_cfg.dir, self.dict_cfg.size,
                                        K, submodule_to_autoencoder, 
                                        patch_vector=patch_vector, inverse=False)["model"]
            model_no_K_diff = model_no_K_out.logits[:, -1, example["clean_answer"]] - \
                              model_no_K_out.logits[:, -1, example["patch_answer"]]
            circuit_no_K_out = run_with_ablated_features(self.model, example["clean_prefix"],
                                                         self.dict_cfg.dir, self.dict_cfg.size,
                                                         list(circuit_features_no_K), submodule_to_autoencoder,
                                                         patch_vector=patch_vector, inverse=True)["model"]
            circuit_no_K_diff = circuit_no_K_out.logits[:, -1, example["clean_answer"]] - \
                                circuit_no_K_out.logits[:, -1, example["patch_answer"]]
            completeness += circuit_no_K_diff / model_no_K_diff
        
        completeness /= num_examples
        completeness_points.append((circuit_no_K_diff.item(), model_no_K_diff.item()))
        return {"mean_completeness": completeness.item(),
                "completeness_points": completeness_points,
                "K": K}
    

    def evaluate_minimality(self, eval_dataset=None, patch_type='zero', K_size=10, sample_size=5):
        if not eval_dataset:    # evaluate on train dataset
            eval_dataset = self.dataset
        circuit_feature_list = self.get_feature_list()
        circuit_feature_set = set(circuit_feature_list)

        if patch_type == "zero":
            patch_vector = t.zeros(self.dict_cfg.size)
        elif patch_type == "mean":
            raise NotImplementedError()
        elif patch_type == "random":
            raise NotImplementedError()

        # Pre-load all submodules and dictionaries
        num_layers = self.model.config.num_hidden_layers
        submodule_to_autoencoder = {}
        for layer in range(num_layers):
            for submodule_name in self.submodules_generic:
                submodule_name = submodule_name.format(str(layer))
                submodule, dictionary = load_submodule_and_dictionary(self.model, submodule_name, self.dict_cfg)
                submodule_to_autoencoder[submodule] = dictionary
        
        num_examples = len(eval_dataset)
        minimality_per_node = {}
        min_minimality = float("inf")
        for node in tqdm(circuit_feature_list, desc="Evaluating minimality", total=len(circuit_feature_list)):
            circuit_features_without_node = deepcopy(circuit_feature_set)
            circuit_features_without_node.remove(node)
            mean_minimality = 0.
            for example in eval_dataset:
                circuit_out = run_with_ablated_features(self.model, example["clean_prefix"],
                                                            self.dict_cfg.dir, self.dict_cfg.size,
                                                            list(circuit_feature_set), submodule_to_autoencoder,
                                                            patch_vector=patch_vector, inverse=True)["model"]
                circuit_diff = circuit_out.logits[:, -1, example["clean_answer"]] - \
                               circuit_out.logits[:, -1, example["patch_answer"]]
                circuit_without_node_out = run_with_ablated_features(self.model, example["clean_prefix"],
                                                            self.dict_cfg.dir, self.dict_cfg.size,
                                                            list(circuit_features_without_node), submodule_to_autoencoder,
                                                            patch_vector=patch_vector, inverse=True)["model"]
                circuit_without_node_diff = circuit_without_node_out.logits[:, -1, example["clean_answer"]] - \
                                            circuit_without_node_out.logits[:, -1, example["patch_answer"]]
                minimality = 1 - (circuit_without_node_diff / circuit_diff)
                mean_minimality += minimality
            mean_minimality /= num_examples
            minimality_per_node[node] = mean_minimality.item() / len(circuit_feature_list)
            min_minimality = min(minimality_per_node[node], min_minimality)

        return {"min_minimality": min_minimality,
                "minimality_per_node": minimality_per_node}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="EleutherAI/pythia-70m-deduped",
                        help="Name of model on which dictionaries were trained.")
    parser.add_argument("--submodules", "-s", type=str, default="model.gpt_neox.layers.{}.attention.dense,model.gpt_neox.layers.{}.mlp.dense_4h_to_h",
                        help="Name of submodules on which dictionaries were trained (with `{}` where the layer number should be).")
    parser.add_argument("--dictionary_dir", "-a", type=str, default="/share/projects/dictionary_circuits/autoencoders/")
    parser.add_argument("--dictionary_size", "-S", type=int, default=32768,
                        help="Width of trained dictionaries.")
    parser.add_argument("--dataset", "-d", type=str, default="/share/projects/dictionary_circuits/data/phenomena/simple.json")
    parser.add_argument("--num_examples", "-n", type=int, default=10,
                        help="Number of example pairs to use in the causal search.")
    parser.add_argument("--threshold", "-t", type=float, default=0.05)
    parser.add_argument("--patch_method", "-p", type=str, choices=["all-folded", "separate", "ig", "exact"],
                        default="separate", help="Method to use for attribution patching.")
    parser.add_argument("--sequence_aggregation", type=str, choices=["final_pos_only", "sum", "max"], default="final_pos_only",)
    parser.add_argument("--chainrule", action="store_true", help="Use chainrule to evaluate circuit.")

    parser.add_argument("--evaluate", action="store_true", help="Load and evaluate a circuit.")
    parser.add_argument("--circuit_path", type=str, default=None, help="Path to circuit to load.")
    parser.add_argument("--seed", type=int, default=12, help="Random seed.")
    args = parser.parse_args()

    submodules = args.submodules
    if "," in submodules:
        submodules = submodules.split(",")
    else:
        submodules = [submodules]

    model = LanguageModel(args.model, dispatch=True)
    model.local_model.requires_grad_(True)
    dataset = load_examples(args.dataset, args.num_examples, model, seed=args.seed, pad_to_length=16)
    dictionary_dir = os.path.join(args.dictionary_dir, args.model.split("/")[-1])

    if args.circuit_path is None:
        save_path = "circuits/"     # save directory
        save_path += os.path.splitext(os.path.basename(args.dataset))[0]   # basename (without extension) of dataset file
        save_path += f"_circuit_threshold{args.threshold}_agg{args.sequence_aggregation.replace('_', '-')}_patch{args.patch_method}_seed{args.seed}"
        if args.chainrule:
            save_path += "_chain"
        save_path += ".pkl"
    else:
        save_path = args.circuit_path
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(save_path)

    circuit = Circuit(model, submodules, dictionary_dir, args.dictionary_size, dataset, threshold=args.threshold)
    if args.evaluate:
        circuit.from_dict(save_path)
        faithfulness = circuit.evaluate_faithfulness()
        print(f"Faithfulness: {faithfulness}")
        completeness = circuit.evaluate_completeness()
        print(f"Completeness: {completeness['mean_completeness']}")
        # minimality = circuit.evaluate_minimality()
        # print(f"Minimality: {minimality['min_minimality']}")
        # print(f"Minimality per node: {minimality['minimality_per_node']}")
    else:
        if args.chainrule:
            circuit.locate_circuit_chainrule(patch_method=args.patch_method, sequence_aggregation=args.sequence_aggregation)
        else:
            circuit.locate_circuit(patch_method=args.patch_method)
        print(circuit.to_dict())
        with open(save_path, 'wb') as pickle_file:
            pickle.dump(circuit.to_dict(), pickle_file)