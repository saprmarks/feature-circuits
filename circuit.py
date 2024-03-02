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
    load_examples, load_submodule_and_dictionary, submodule_name_to_type_layer, DictionaryCfg, load_submodules_and_dictionaries
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
        method='all-folded',
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
    
    # alternative caching gradients for all upstream nodes at once
    for downstream_idx, downstream_submod in reversed(list(enumerate(submodules))):
        upstream_submods = submodules[:downstream_idx]
        upstream_dictionaries = [dictionaries[upstream_submod] for upstream_submod in upstream_submods]
        upstream_grads = get_grad(
            clean,
            patch,
            model,
            dictionaries,
            upstream_submods,
            downstream_submod,
            feat_idxs[downstream_submod],
        )
        for downstream_node in nodes_by_component[downstream_submod]:
            for upstream_submod in upstream_submods:
                for upstream_node in nodes_by_component[upstream_submod]:
                    grads[(downstream_node, upstream_node)] = upstream_grads[downstream_node.feat_idx][upstream_submod][tuple(upstream_node.feat_idx)]


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
    
def split_dataset(dataset):
    clean_inputs = t.cat([example['clean_prefix'] for example in dataset], dim=0)
    patch_inputs = t.cat([example['patch_prefix'] for example in dataset], dim=0)
    final_token_positions = t.tensor([example['prefix_length_wo_pad'] for example in dataset]).int() # token position before padding, 1-indexed
    clean_answer_idxs, patch_answer_idxs = [], []
    for example in dataset:
        clean_answer_idxs.append(example['clean_answer'])
        patch_answer_idxs.append(example['patch_answer'])
    clean_answer_idxs = t.Tensor(clean_answer_idxs).long()
    patch_answer_idxs = t.Tensor(patch_answer_idxs).long()
    print(clean_inputs.shape, patch_inputs.shape, clean_answer_idxs.shape, patch_answer_idxs.shape, final_token_positions.shape)
    return clean_inputs, patch_inputs, clean_answer_idxs, patch_answer_idxs, final_token_positions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, help="Name of model on which dictionaries were trained.",
                        default="EleutherAI/pythia-70m-deduped")
    parser.add_argument("--use_attn", type=bool, help="Whether to include attention features.",
                        default=False) # TODO: Using attn and MLP simultaneously is not supported, bc they are passed in submodule array in hierarchy (i.e. the algorithm assumes output of attn is the innput to MLP in the same layer), but they run in parallel so have no gradients onto each other. We have to implement layer hierarchy (insetead of submodule hierarchy) across this scircuit discovery method.
    parser.add_argument("--use_mlp", type=bool, help="Whether to include MLP features.",
                        default=True)
    parser.add_argument("--use_resid", type=bool, help="Whether to include residual stream features.",
                        default=True)
    parser.add_argument("--dictionary_dir", "-a", type=str,
                        default="/share/projects/dictionary_circuits/autoencoders/")
    parser.add_argument("--dictionary_size", "-S", type=int, help="Width of trained dictionaries.",
                        default=32768)
    parser.add_argument("--dataset", "-d", type=str,
                        default="/share/projects/dictionary_circuits/data/phenomena/simple.json")
    parser.add_argument("--num_examples", "-n", type=int, help="Number of example pairs to use in the causal search.",
                        default=1)
    parser.add_argument("--node_threshold", "-nt", type=float, help="Threshold for node inclusion.",
                        default=0.1)
    parser.add_argument("--edge_threshold", "-et", type=float, help="Threshold for edge inclusion.",
                        default=0.01)
    parser.add_argument("--patch_method", "-p", type=str, help="Method to use for attribution patching.",
                        default="separate", choices=["all-folded", "separate", "ig", "exact"])
    parser.add_argument("--sequence_aggregation", type=str, 
                        default="final_pos_only", choices=["final_pos_only", "sum", "max"])
    parser.add_argument("--seed", type=int, help="Random seed.",
                        default=12)
    parser.add_argument("--device", type=str, help="Device to use for computation.",
                        default="cuda:0")
    parser.add_argument("--pad_to_length", type=int, help="Pad examples to this length.",
                        default=16)
    parser.add_argument("--circuit_path", type=str, help="Path to save circuit to.",
                        default=None)
    # Arguments for circuit evaluation which have been moved to circuit_evaluation.py
    # parser.add_argument("--evaluate", action="store_true", help="Load and evaluate a circuit.") # extra file
    # parser.add_argument("--circuit_path", type=str, default=None, help="Path to circuit to load.") # extra file
    args = parser.parse_args()

    # Create directory to save circuit
    if args.circuit_path is None:
        save_path = "/home/can/dictionary-circuits/circuits/"     # save directory
        save_path += os.path.splitext(os.path.basename(args.dataset))[0]   # basename (without extension) of dataset file
        save_path += f"_circuit_tnode{args.node_threshold}_tedge{args.edge_threshold}_agg{args.sequence_aggregation.replace('_', '-')}_patch{args.patch_method}_seed{args.seed}"
        save_path += ".pkl"
    else:
        save_path = args.circuit_path
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(save_path)

    # Load model, dataset, and dictionaries
    model = LanguageModel(args.model, device_map=args.device)
    dataset = load_examples(args.dataset, args.num_examples, model, seed=args.seed, pad_to_length=args.pad_to_length)
    dictionary_dir = os.path.join(args.dictionary_dir, args.model.split("/")[-1])
    submodules, submodule_names, dictionaries = load_submodules_and_dictionaries(
        model, use_attn=args.use_attn, use_mlp=args.use_mlp, use_resid=args.use_resid,
        dict_path=dictionary_dir, dict_size=args.dictionary_size, device=args.device)

    # # Simple test case
    # clean_idx = model.tokenizer(" is").input_ids[-1]
    # patch_idx = model.tokenizer(" are").input_ids[-1]

    # # Define metric: logit diff patch-clean
    # def metric_fn(model):
    #     return model.embed_out.output[:,-1,patch_idx] - model.embed_out.output[:,-1,clean_idx]

    clean_inputs, patch_inputs, clean_answer_idxs, patch_answer_idxs, final_token_positions = split_dataset(dataset)

    def metric_fn(model):
        indices_batch_dim = t.arange(clean_answer_idxs.shape[0])
        logits = model.embed_out.output[indices_batch_dim, final_token_positions-1, :]
        logit_diff = t.gather(
            logits, dim=-1, index=patch_answer_idxs.unsqueeze(-1)
        ) - t.gather(
            logits, dim=-1, index=clean_answer_idxs.unsqueeze(-1)
        )
        return logit_diff.squeeze(-1)

    # Run circuit discovery
    dag = get_circuit(
        clean_inputs, # ["The man", "The tall boy"],
        patch_inputs, # ["The men", "The tall boys"],
        model,
        submodules,
        submodule_names,
        dictionaries,
        metric_fn,
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
        method=args.patch_method
    )

    print(dag)

    # TODO: Solve CUDA out of memory error, that occurs when setting --num_examples to 10

    # TODO: Using attn and MLP simultaneously is not supported, bc they are passed in submodule array in hierarchy (i.e. the algorithm assumes output of attn is the innput to MLP in the same layer), but they run in parallel so have no gradients onto each other. We have to implement layer hierarchy (insetead of submodule hierarchy) across this scircuit discovery method.

    # TODO: Connect Sam's implementation of sliced, mean, sum DAGs to the sequence aggregation argument in the argparser

    # TODO: Convert DAG to a lightweight object which enables pickling
    # with open(save_path, 'wb') as pickle_file:
    #     pickle.dump(dag, pickle_file)

    # TODO: Adapt circuit evaluation in circuit_evaluation.py to use the lightweight object