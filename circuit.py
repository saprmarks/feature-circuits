import argparse
import os
import pickle
import torch as t

from nnsight import LanguageModel
from tqdm import tqdm
from collections import defaultdict
from dictionary_learning.buffer import ActivationBuffer
from dictionary_learning.dictionary import AutoEncoder
from causal_search import (
    load_examples, load_submodule, submodule_type_to_name
)
from acdc import patching_on_y, patching_on_downstream_feature
from subnetwork import (
    Node, Subnetwork,
)
from patching import subnetwork_patch


class CircuitNode:
    def __init__(self, name, data = None, children = None, parents = None):
        self.name = name    # format: `{layer}_{idx}` OR `y`
        self.data = data    # TODO: 10 sentences w/ activated tokens?
        if not children:
            self.children = []
        else:
            self.children = children
        if not parents:
            self.parents = []
        else:
            self.parents = parents
        
        self.effect_on_parent = None

    def add_child(self, child):
        if "_" in self.name:
            this_layer = self.name.split("_")[0]
            child_layer = child.name.split("_")[0]
            if child_layer >= this_layer:
                raise Exception(f"Invalid child: {self.name} -> {child.name}")
        self.children.append(child)
        child.parents.append(self)

    def __eq__(self, other):
        return self.name == other.name
    
    def __hash__(self):
        return hash(self.name)


class Circuit:
    def __init__(self, model, submodules, dictionary_dir, dictionary_size, dataset):
        self.model = model
        self.submodules_generic = submodules
        self.dictionary_dir = dictionary_dir
        self.dictionary_size = dictionary_size
        self.dataset = dataset
        self.y_threshold = 0.025
        self.feat_threshold = 0.1
        self.path_threshold = 0.01
        self.filter_proportion = 0.25

        self.root = CircuitNode("y")
    
    def load_dictionary(self, layer, submodule, submodule_type):
        dict_id = "1" if submodule_type == "mlp" else "0"
        dict_path = os.path.join(self.dictionary_dir,
                                 f"{submodule_type}_out_layer{layer}",
                                 f"{dict_id}_{self.dictionary_size}/ae.pt")
        try:
            submodule_width = submodule.out_features
        except AttributeError:
            # is residual. need to load model to get this
            with self.model.invoke("test") as invoker:
                hidden_states = submodule.output.save()
            hidden_states = hidden_states.value
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
            submodule_width = hidden_states.shape[2]
        autoencoder = AutoEncoder(submodule_width, self.dictionary_size).cuda()
        # TODO: add support for both of these cases to the `load_state_dict` method
        try:
            autoencoder.load_state_dict(t.load(dict_path))
        except TypeError:
            autoencoder.load_state_dict(t.load(dict_path).state_dict())
        return autoencoder
    

    def _get_paths_to_root(self, lower_node, upper_node):
        for parent in upper_node.parents:
            if parent.name == "y":
                yield [lower_node, upper_node]
            else:
                for path in self._get_paths_to_root(upper_node, parent):
                    yield [lower_node] + path

    """
    # TODO: test
    def path_patching(self, lower_node, upper_node):
        # Returns indirect effect on logits via existing path in circuit.
        def _get_paths_to_root(lower_node, upper_node):
            if len(upper_node.parents) == 1:
                yield [lower_node, upper_node]
            for parent in upper_node.parents:
                if parent == "y":
                    yield [lower_node, upper_node]
                else:
                    yield [lower_node] + _get_paths_to_root(upper_node, parent)
        
        paths = _get_paths_to_root(lower_node, upper_node)
        for path in paths:
            layer_list = [node.name.split("_")[0] for node in path]
            lowest_layer = layer_list[0]
            top_layer = layer_list[-1]
            for example_idx, example in enumerate(self.dataset):
                dictionary_activations = get_submodule_activation_caches(self.model, submodule,
                                                example["clean_prefix"], example["patch_prefix"])

                # get clean logits
                with model.forward(example["clean_prefix"]) as invoker_clean:
                    pass    # no interventions necessary
    """


    def locate_circuit(self):
        num_layers = self.model.config.num_hidden_layers
        nodes_per_layer = defaultdict(list)
        # Iterate backwards through layers. Establish causal effects
        for layer_i in tqdm(range(num_layers-1, -1, -1), desc="Layer",
                            total=num_layers):
            # First, get effect on output y
            submodules_i_name = [submodule.format(str(layer_i)) for submodule in self.submodules_generic]
            submodules_i_type = ["mlp" if "mlp" in s else "attn" if "attention" in s else "resid" for s in submodules_i_name]
            submodules_i = [load_submodule(self.model, submodule_i_name) for submodule_i_name in submodules_i_name]
            dictionaries_i = [self.load_dictionary(layer_i, submodules_i[idx], submodules_i_type[idx]) for idx in range(len(submodules_i))]
            effects_on_y = patching_on_y(self.dataset, self.model, submodules_i, dictionaries_i,
                                         method="separate")
            effects_on_y = effects_on_y.effects

            # if effect greater than threshold, add to graph
            for submodule_idx, submodule in enumerate(effects_on_y):
                feature_indices = (effects_on_y[submodule][-1, :] > self.y_threshold).nonzero().flatten().tolist()
                submodule_type = submodules_i_type[submodule_idx]
                for feature_idx in feature_indices:
                    node_name = f"{layer_i}_{feature_idx}_{submodule_type}"
                    child = CircuitNode(node_name)
                    child.effect_on_parent = effects_on_y[submodule][-1, feature_idx].item()
                    nodes_per_layer[layer_i].append(child)
                    self.root.add_child(child)

            # Second, get effect on other features already in the graph (above this feature)
            # TODO: test
            for layer_j in tqdm(range(num_layers-1, layer_i, -1), leave=False, desc="Layers above",
                                total = num_layers - layer_i):
                # causal effect of lower features i on upper features j
                candidate_paths = []
                # first, filter possible features i for those that actually change the activation of feature j
                for node_j in nodes_per_layer[layer_j]:
                    submodule_j_type = "mlp" if "mlp" in node_j.name else "attn" if "attn" in node_j.name else "resid"
                    submodule_j_name = submodule_type_to_name(submodule_j_type).format(layer_j)
                    submodule_j = load_submodule(self.model, submodule_j_name)
                    dictionary_j = self.load_dictionary(layer_j, submodule_j, submodule_j_type)
                    feat_idx_j = int(node_j.name.split("_")[1])
                    effects_on_feat_j = patching_on_downstream_feature(self.dataset, self.model, submodules_i, dictionaries_i,
                                            submodule_j, dictionary_j, downstream_feature_id=feat_idx_j,
                                            method='separate')
                    effects_on_feat_j = effects_on_feat_j.effects

                    for submodule_idx, submodule in enumerate(effects_on_feat_j):
                        feature_indices = (effects_on_feat_j[submodule][-1, :] > self.feat_threshold).nonzero().flatten().tolist()
                        submodule_type = submodules_i_type[submodule_idx]
                        for feature_idx in feature_indices:
                            node_name = f"{layer_i}_{feature_idx}_{submodule_type}"
                            node_i = CircuitNode(node_name)
                            node_i.effect_on_parent = effects_on_feat_j[submodule][-1, feature_idx].item()
                            nodes_per_layer[layer_i].append(node_i)
                            node_j.add_child(node_i)
                            """
                            # add this back in
                            # find all possible paths from node_i to node_j to root
                            candidate_paths_i_j = self._get_paths_to_root(node_i, node_j)
                            for candidate_path in candidate_paths_i_j:
                                candidate_paths.append(candidate_path)
                            """
                """
                # add this back in
                # now, iterate through candidate paths and keep those that effect the output above the threshold
                for candidate_path in candidate_paths:
                    # build subnetwork
                    # TODO: build submodules list, build autoencoders list
                    path_submodules = []
                    path_autoencoders = []
                    subnetwork = Subnetwork()
                    for idx, node in enumerate(candidate_path):
                        layer, feat_idx, submodule_type = node.name.split("_")
                        submodule_suffix = ""   # residual
                        if submodule_type == "mlp":
                            submodule_suffix = ".mlp.dense_4h_to_h"
                        elif submodule_type == "attn":
                            submodule_suffix = ".attention.dense"
                        submodule_node = Node(self.model, f'gpt_neox.layers.{layer}{submodule_suffix}/{feat_idx}')
                        subnetwork.add_node(submodule_node)
                        # first (lowest) node in subcircuit
                        if idx == 0:
                            start_node = submodule_node
                        # load submodule and autoencoder
                        path_submodule_name = submodule_type_to_name(submodule_j_type).format(layer_j)
                        path_submodule = load_submodule(self.model, submodule_j_name)
                        path_dictionary = self.load_dictionary(layer_j, submodule_j, submodule_j_type)
                        path_submodules.append(path_submodule)
                        path_autoencoders.append(path_dictionary)
                    path_effect = subnetwork_patch(self.dataset, self.model, path_submodules, path_autoencoders,
                                                   subnetwork, start_node)

                    if path_effect > self.path_threshold:
                        child = CircuitNode(candidate_path[0])
                        parent = CircuitNode(candidate_path[1])
                        parent.add_child(child)
                        child.effect_on_parent = path_effect
                        nodes_per_layer[layer_i].append(child)
                """
                        
    def to_dict(self):
        # Depth-first search
        def _dfs(curr_node, d = {}):
            d[curr_node.name] = []
            # if children, go through each
            for child in curr_node.children:
                d[curr_node.name].append((child.name, child.effect_on_parent))
                _dfs(child, d=d)
            # else, return dictionary
            return d

        out_dict = _dfs(self.root)
        return out_dict
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="EleutherAI/pythia-70m-deduped",
                        help="Name of model on which dictionaries were trained.")
    parser.add_argument("--submodules", "-s", type=str, default="model.gpt_neox.layers.{}.mlp.dense_4h_to_h",
                        help="Name of submodules on which dictionaries were trained (with `{}` where the layer number should be).")
    parser.add_argument("--dictionary_dir", "-a", type=str, default="/share/projects/dictionary_circuits/autoencoders/")
    parser.add_argument("--dictionary_size", "-S", type=int, default=32768,
                        help="Width of trained dictionaries.")
    parser.add_argument("--dataset", "-d", type=str, default="/share/projects/dictionary_circuits/data/phenomena/simple.json")
    parser.add_argument("--num_examples", "-n", type=int, default=100,
                        help="Number of example pairs to use in the causal search.")
    args = parser.parse_args()

    submodules = args.submodules
    if "," in submodules:
        submodules = submodules.split(",")
    else:
        submodules = [submodules]

    model = LanguageModel(args.model, dispatch=True)
    model.local_model.requires_grad_(True)
    dataset = load_examples(args.dataset, args.num_examples, model)
    dictionary_dir = os.path.join(args.dictionary_dir, args.model.split("/")[-1])

    circuit = Circuit(model, submodules, dictionary_dir, args.dictionary_size, dataset)
    circuit.locate_circuit()
    print(circuit.to_dict())

    save_path = args.dataset.split("/")[-1].split(".json")[0] + "_old_circuit.pkl"
    with open(save_path, 'wb') as pickle_file:
        pickle.dump(circuit.to_dict(), pickle_file)