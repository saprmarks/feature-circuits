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
        return self.name == other.name and \
            self.submodule == other.submodule and \
                self.dictionary == other.dictionary and \
                    (self.feat_idx == other.feat_idx).all()

def get_circuit(
        clean, 
        patch,
        model,
        submodules,
        dictionaries,
        metric_fn,
        node_threshold=0.1,
        edge_threshold=0.05,
        method='separate',
):
    
    # first, figure out which features get to go in the circuit and the gradient of y wrt these feeatures
    effects, deltas, grads, _ = patching_effect(clean, patch, model, submodules, dictionaries, metric_fn, method=method)
    feat_idxs = {}
    grads_to_y = {}
    for submodule, effect in effects.items():
        feat_idxs[submodule] = t.nonzero(effect > node_threshold)
        grads_to_y[submodule] = TensorDict({idx : grads[submodule][tuple(idx)] for idx in feat_idxs[submodule]})

    # construct DAG
    dag = WeightedDAG()
    grads = {} # stores grads between any two pairs of nodes

    y = CircuitNode("y", None, None, None)
    dag.add_node(y)

    nodes_by_component = {}
    for component_idx, (submodule, idxs) in reversed(list(enumerate(feat_idxs.items()))): # assuming that submodules are properly ordered
        # get nodes for this component
        nodes = []
        for idx in idxs:
            n = CircuitNode(f'{component_idx}/{idx}', submodule, dictionaries[component_idx], idx)
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
        for upstream_idx, upstream_submod in enumerate(submodules[:downstream_idx]):
            upstream_grads = get_grad(
                clean,
                patch,
                model,
                upstream_submod,
                dictionaries[upstream_idx],
                downstream_submod,
                dictionaries[downstream_idx],
                feat_idxs[downstream_submod],
            )
            for downstream_node in nodes_by_component[downstream_submod]:
                for upstream_node in nodes_by_component[upstream_submod]:
                    grads[(downstream_node, upstream_node)] = upstream_grads[downstream_node.feat_idx][tuple(upstream_node.feat_idx)]

    # compute the edge weights
    dag = deduce_edge_weights(dag, grads)

    # reassign weights to represent attribution scores
    for n1, n2 in dag.edges:
        dag.add_edge(n1, n2, weight=dag.node_weight(n1) * dag.edge_weight((n1, n2)))

    # filter edges
    for n1, n2 in dag.edges:
        if dag.edge_weight((n1, n2)) < edge_threshold:
            dag.remove_edge((n1, n2))

    return dag
