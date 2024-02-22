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

if __name__ == "__main__":
    from dictionary_learning import AutoEncoder

    model = LanguageModel('EleutherAI/pythia-70m-deduped', device_map='cuda:0')
    submodules = []
    submodule_names = {}
    dictionaries = {}
    for layer in range(len(model.gpt_neox.layers)):
        submodule = model.gpt_neox.layers[layer].mlp
        submodule_names[submodule] = f'mlp{layer}'
        submodules.append(submodule)
        ae = AutoEncoder(512, 64 * 512).cuda()
        ae.load_state_dict(t.load(f'/share/projects/dictionary_circuits/autoencoders/pythia-70m-deduped/mlp_out_layer{layer}/5_32768/ae.pt'))
        dictionaries[submodule] = ae

        submodule = model.gpt_neox.layers[layer]
        submodule_names[submodule] = f'resid{layer}'
        submodules.append(submodule)
        ae = AutoEncoder(512, 64 * 512).cuda()
        ae.load_state_dict(t.load(f'/share/projects/dictionary_circuits/autoencoders/pythia-70m-deduped/resid_out_layer{layer}/5_32768/ae.pt'))
        ae.load_state_dict(t.load(f'/share/projects/dictionary_circuits/autoencoders/pythia-70m-deduped/resid_out_layer{layer}/5_32768/ae.pt'))
        dictionaries[submodule] = ae

    clean_idx = model.tokenizer(" is").input_ids[-1]
    patch_idx = model.tokenizer(" are").input_ids[-1]

    def metric_fn(model):
        return model.embed_out.output[:,-1,patch_idx] - model.embed_out.output[:,-1,clean_idx]

    dag = get_circuit(
        ["The man", "The tall boy"],
        ["The men", "The tall boys"],
        model,
        submodules,
        submodule_names,
        dictionaries,
        metric_fn,
    )

    reduced_dag = mean_dag(sum_dag(dag, 1), 0, crosses=False)