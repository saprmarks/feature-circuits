import argparse
import os
import pickle
import random
import torch as t
from graph_utils import WeightedDAG, deduce_edge_weights
from attribution import patching_effect, get_grad
from tensordict import TensorDict
from dictionary_learning import AutoEncoder

from nnsight import LanguageModel
from tqdm import tqdm, trange
from collections import defaultdict
from loading_utils import (
    load_examples, load_submodule_and_dictionary, submodule_name_to_type_layer, DictionaryCfg
)
from acdc import patching_on_y, patching_on_downstream_feature
from ablation_utils import run_with_ablated_features
import json

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
    for submodule, idxs in reversed(feat_idxs.items()): # assuming that submodules are properly ordered
        # get nodes for this component
        nodes = []
        for idx in idxs:
            n = CircuitNode(f'{submodule_names[submodule]}/{idx.tolist()}', submodule, dictionaries[submodule], idx)
            nodes.append(n)
        nodes_by_component[submodule] = nodes

        # add nodes to the DAG
        old_nodes = dag.nodes.copy()
        for n in nodes:
            dag.add_node(n, weight=deltas[submodule][tuple(n.feat_idx)])
            grads[(n, y)] = grads_to_y[submodule][n.feat_idx]
            for n_old in old_nodes:
                if n_old == y or n_old.feat_idx[0] == n.feat_idx[0]:
                    dag.add_edge(n, n_old)

    # alternative caching gradients for all upstream nodes at once
    for downstream_idx, downstream_submod in reversed(list(enumerate(submodules))):
        upstream_submods = submodules[:downstream_idx]
        upstream_grads = get_grad(
            clean,
            patch,
            model,
            dictionaries,
            upstream_submods,
            downstream_submod,
            feat_idxs[downstream_submod],
            return_idxs = {submod : [n.feat_idx for n in nodes_by_component[submod]] for submod in upstream_submods}
        )
        for downstream_node in nodes_by_component[downstream_submod]:
            for upstream_submod in upstream_submods:
                for upstream_node in nodes_by_component[upstream_submod]:
                    grads[(upstream_node, downstream_node)] = upstream_grads[downstream_node.feat_idx][upstream_submod][upstream_node.feat_idx].item()

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
        if abs(dag.edge_weight((n1, n2))) < edge_threshold:
            dag.remove_edge((n1, n2))

    return dag

def slice_dag(dag, pos, dim):
    """
    Given a DAG whose nodes are CircuitNodes, returns a new DAG consisting only of those nodes whose feat_idx[dim] == pos
    """
    if type(pos) == list:
        new_dag = WeightedDAG()
        for n in dag.nodes:
            if n.feat_idx is None or n.feat_idx[dim] == pos[n.feat_idx[0]]:
                new_dag.add_node(n, weight=dag.node_weight(n))
        for n1, n2 in dag.edges:
            if n1 in new_dag.nodes and n2 in new_dag.nodes:
                new_dag.add_edge(n1, n2, weight=dag.edge_weight((n1, n2)))
        return new_dag
    elif type(pos) == int:
        new_dag = WeightedDAG()
        for n in dag.nodes:
            if n.feat_idx is None or n.feat_idx[dim] == pos:
                new_dag.add_node(n, weight=dag.node_weight(n))
        for n1, n2 in dag.edges:
            if n1 in new_dag.nodes and n2 in new_dag.nodes:
                new_dag.add_edge(n1, n2, weight=dag.edge_weight((n1, n2)))
        return new_dag
    else:
        raise ValueError("pos should be an int or a list of ints")
        

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
        name = f"{'/'.join(n.name.split('/')[:-1])}/{feat_idx.tolist()}"
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
            name = f"{'/'.join(n.name.split('/')[:-1])}/{feat_idx.tolist()}"
            n_summed = CircuitNode(name, n.submodule, n.dictionary, feat_idx)
            if new_dag.has_node(n_summed):
                new_dag.add_node(n_summed, weight=new_dag.node_weight(n_summed) + dag.node_weight(n))
            else:
                new_dag.add_node(n_summed, weight=dag.node_weight(n))
    for e in dag.edges:
        n1, n2 = e
        if n2.name == 'y':
            n1_summed_name = f"{'/'.join(n1.name.split('/')[:-1])}/{_remove_coord(n1.feat_idx, dim).tolist()}"
            n1_summed = CircuitNode(n1_summed_name, n1.submodule, n1.dictionary, _remove_coord(n1.feat_idx, dim))
            n2_summed = n2
        elif n1.feat_idx[dim] != n2.feat_idx[dim]:
            raise ValueError(f"crosses was set to False, but some edge crosses positions in dimension {dim}")
        else:
            n1_summed_name = f"{'/'.join(n1.name.split('/')[:-1])}/{_remove_coord(n1.feat_idx, dim).tolist()}"
            n2_summed_name = f"{'/'.join(n2.name.split('/')[:-1])}/{_remove_coord(n2.feat_idx, dim).tolist()}"
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
        name = f"{'/'.join(n.name.split('/')[:-1])}/{feat_idx.tolist()}"
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="EleutherAI/pythia-70m-deduped",
                        help="Name of model on which dictionaries were trained.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument('--submodules', type=str, nargs='+', default=['attn', 'mlp', 'resid'])
    parser.add_argument('--dicts', type=str, default='5_32768')
    parser.add_argument('--activation_dim', type=int, default=512, help='Dimension of the activation space.')
    # parser.add_argument("--submodules", "-s", type=str, nargs='+', default="model.gpt_neox.layers.{}.attention.dense,model.gpt_neox.layers.{}.mlp.dense_4h_to_h",
    #                     help="Name of submodules on which dictionaries were trained (with `{}` where the layer number should be).")
    # parser.add_argument("--dictionary_dir", "-a", type=str, default="/share/projects/dictionary_circuits/autoencoders/")
    # parser.add_argument("--dictionary_size", "-S", type=int, default=32768,
    #                     help="Width of trained dictionaries.")
    parser.add_argument("--dataset", "-d", type=str, default="/share/projects/dictionary_circuits/data/phenomena/simple.json")
    parser.add_argument("--num_examples", "-n", type=int, default=10,
                        help="Number of example pairs to use in the causal search.")
    parser.add_argument("--node_threshold", type=float, default=0.1)
    parser.add_argument("--edge_threshold", type=float, default=0.01)
    parser.add_argument("--patch_method", "-p", type=str, choices=["all-folded", "separate", "ig", "exact"],
                        default="separate", help="Method to use for attribution patching.")
    parser.add_argument("--sequence_aggregation", type=str, choices=["final_pos_only", "none", "sum", "max"], default="final_pos_only",)
    parser.add_argument("--chainrule", action="store_true", help="Use chainrule to evaluate circuit.")

    parser.add_argument("--evaluate", action="store_true", help="Load and evaluate a circuit.")
    parser.add_argument("--circuit_path", type=str, default=None, help="Path to circuit to load.")
    parser.add_argument("--seed", type=int, default=12, help="Random seed.")
    args = parser.parse_args()


    model = LanguageModel(args.model, device_map=args.device)
    submodules = []
    submodule_names, dictionaries = {},{}
    for layer in range(len(model.gpt_neox.layers)):
        if 'attn' in args.submodules:
            submodule = model.gpt_neox.layers[layer].attention
            ae = AutoEncoder(args.activation_dim, 64 * args.activation_dim).to(args.device)
            ae.load_state_dict(t.load(f'/share/projects/dictionary_circuits/autoencoders/pythia-70m-deduped/attn_out_layer{layer}/{args.dicts}/ae.pt'))
            submodules.append(submodule)
            submodule_names[submodule] = f'attn_{layer}'
            dictionaries[submodule] = ae
        if 'mlp' in args.submodules:
            submodule = model.gpt_neox.layers[layer].mlp
            ae = AutoEncoder(args.activation_dim, 64 * args.activation_dim).to(args.device)
            ae.load_state_dict(t.load(f'/share/projects/dictionary_circuits/autoencoders/pythia-70m-deduped/mlp_out_layer{layer}/{args.dicts}/ae.pt'))
            submodules.append(submodule)
            submodule_names[submodule] = f'mlp_{layer}'
            dictionaries[submodule] = ae
        if 'resid' in args.submodules:
            submodule = model.gpt_neox.layers[layer]
            ae = AutoEncoder(args.activation_dim, 64 * args.activation_dim).to(args.device)
            ae.load_state_dict(t.load(f'/share/projects/dictionary_circuits/autoencoders/pythia-70m-deduped/resid_out_layer{layer}/{args.dicts}/ae.pt'))
            submodules.append(submodule)
            submodule_names[submodule] = f'resid_{layer}'
            dictionaries[submodule] = ae

    dataset_items = open(args.dataset).readlines()
    random.seed(args.seed)
    random.shuffle(dataset_items)

    examples = load_examples(args.dataset, args.num_examples, model)
    clean_inputs = t.cat([e['clean_prefix'] for e in examples], dim=0)
    patch_inputs = t.cat([e['patch_prefix'] for e in examples], dim=0)
    clean_answer_idxs = t.tensor([e['clean_answer'] for e in examples], dtype=t.long)
    patch_answer_idxs = t.tensor([e['patch_answer'] for e in examples], dtype=t.long)

    def metric_fn(model):
        return (
            t.gather(model.embed_out.output[:,-1,:], dim=-1, index=patch_answer_idxs.view(-1, 1)).squeeze(-1) - \
            t.gather(model.embed_out.output[:,-1,:], dim=-1, index=clean_answer_idxs.view(-1, 1)).squeeze(-1)
        )
    
    dag = get_circuit(
        clean_inputs,
        patch_inputs,
        model,
        submodules,
        submodule_names,
        dictionaries,
        metric_fn,
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
    )

    if args.sequence_aggregation == "sum":
        dag = sum_dag(dag, dim=1, crosses=True)
    elif args.sequence_aggregation == "none":
        pass
    else:
        raise NotImplementedError("Only sum is implemented for sequence_aggregation")

    dag = mean_dag(dag, dim=0, crosses=False)

    # filter out nodes
    for n in dag.nodes:
        if n.name == 'y':
            continue
        try:
            if dag.node_weight(n) < args.node_threshold:
                dag.remove_node(n)
        except:
            print(n, dag.node_weight(n))
            assert False
    
    # filter out edges
    for n1, n2 in dag.edges:
        if abs(dag.edge_weight((n1, n2))) < args.edge_threshold:
            dag.remove_edge((n1, n2))


    edge_dict = {
        str(n1) : {
            str(n2) : w.item() for n2, w in dag._edges[n1].items()
        } for n1 in dag._edges
    }
    node_dict = {
        str(n) : w.sum().item() for n, w in dag._nodes.items() # TODO get rid of sum later, only useful if not aggregating
    }
    with open(f'test_circuit{args.num_examples}.pkl', 'wb') as f:
        pickle.dump((edge_dict, node_dict), f)



    # dataset = load_examples(args.dataset, args.num_examples, model, seed=args.seed, pad_to_length=16)
    # dictionary_dir = os.path.join(args.dictionary_dir, args.model.split("/")[-1])

    # if args.circuit_path is None:
    #     save_path = "circuits/"     # save directory
    #     save_path += os.path.splitext(os.path.basename(args.dataset))[0]   # basename (without extension) of dataset file
    #     save_path += f"_circuit_threshold{args.threshold}_agg{args.sequence_aggregation.replace('_', '-')}_patch{args.patch_method}_seed{args.seed}"
    #     if args.chainrule:
    #         save_path += "_chain"
    #     if args.nopair:
    #         save_path += "_nopair_logprob"
    #     save_path += ".pkl"
    # else:
    #     save_path = args.circuit_path
    # if not os.path.exists(os.path.dirname(save_path)):
    #     os.makedirs(save_path)

    # circuit = Circuit(model, submodules, dictionary_dir, args.dictionary_size, dataset, threshold=args.threshold)
    # if args.evaluate:
    #     circuit.from_dict(save_path)
    #     faithfulness = circuit.evaluate_faithfulness()
    #     print(f"Faithfulness: {faithfulness}")
    #     completeness = circuit.evaluate_completeness()
    #     print(f"Completeness: {completeness['mean_completeness']}")
    #     # minimality = circuit.evaluate_minimality()
    #     # print(f"Minimality: {minimality['min_minimality']}")
    #     # print(f"Minimality per node: {minimality['minimality_per_node']}")
    # else:
    #     if args.chainrule:
    #         circuit.locate_circuit_chainrule(patch_method=args.patch_method, sequence_aggregation=args.sequence_aggregation, nopair=args.nopair)
    #     else:
    #         circuit.locate_circuit(patch_method=args.patch_method)
    #     print(circuit.to_dict())
    #     with open(save_path, 'wb') as pickle_file:
    #         pickle.dump(circuit.to_dict(), pickle_file)