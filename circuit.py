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
    load_examples, load_submodule, relative_prob_change, logit_diff,
    attribution_patching, attribution_patching_wrt_features
)
from acdc import patching_on_y


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
    def __init__(self, model, submodules, dictionary_dir, dictionary_size, dataset,
                 metric_fn = relative_prob_change):
        self.model = model
        self.submodules_generic = submodules
        self.dictionary_dir = dictionary_dir
        self.dictionary_size = dictionary_size
        self.dataset = dataset
        self.metric_fn = metric_fn
        self.y_threshold = 0.2
        self.feat_threshold = 0.2

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


    # TODO: test   
    def locate_circuit(self):
        num_layers = self.model.config.num_hidden_layers
        nodes_per_layer = defaultdict(list)
        # Iterate backwards through layers. Establish causal effects
        # TODO: change upper limit to num_layers-1
        for layer_i in tqdm(range(num_layers-1, -1, -1), desc="Layer",
                            total=num_layers):
            # First, get effect on output y
            submodules_i_name = [submodule.format(str(layer_i)) for submodule in self.submodules_generic]
            submodules_i_type = ["mlp" if "mlp" in s else "attn" if "attention" in s else "resid" for s in submodules_i_name]
            submodules_i = [load_submodule(self.model, submodule_i_name) for submodule_i_name in submodules_i_name]
            dictionaries_i = [self.load_dictionary(layer_i, submodules_i[idx], submodules_i_type[idx]) for idx in range(len(submodules_i))]
            effects_on_y = patching_on_y(self.dataset, self.model, submodules_i, dictionaries_i)

            # if effect greater than threshold, add to graph
            for submodule_idx in range(len(effects_on_y)):
                # print(t.topk(effects_on_y[submodule_idx], 5))
                feature_indices = (effects_on_y[submodule_idx] > self.y_threshold).nonzero().flatten().tolist()
                submodule_type = submodules_i_type[submodule_idx]
                for feature_idx in feature_indices:
                    node_name = f"{layer_i}_{feature_idx}_{submodule_type}"
                    child = CircuitNode(node_name)
                    child.effect_on_parent = effects_on_y[submodule_idx][feature_idx].item()
                    nodes_per_layer[layer_i].append(child)
                    self.root.add_child(child)

            """
            # Second, get effect on other features already in the graph (above this feature)
            # TODO: test
            for layer_j in tqdm(range(num_layers-1, layer_i, -1), leave=False, desc="Layers above",
                                total = num_layers - layer_i):
                # TODO: causal effect of lower features i on upper features j
                submodule_j_name = self.submodule_generic.format(str(layer_j))
                submodule_j = load_submodule(self.model, submodule_j_name)
                dictionary_j = self.load_dictionary(layer_j, submodule_j)
                for node_j in nodes_per_layer[layer_j]:
                    feat_idx_j = node_j.name.split("_")[1]
                    effect_on_feat_j = attribution_patching_wrt_features(self.model, submodule_i, submodule_j,
                                            dictionary_i, dictionary_j, feat_idx_j, self.dataset)
                    indices = (abs(effect_on_feat_j) > self.feat_threshold).nonzero().flatten().tolist()
                    for index in indices:
                        node_name = f"{layer_i}_{index}"
                        child = CircuitNode(node_name)
                        effect_via_path = path_patching(child, node_j)
                        if child not in nodes_per_layer[layer_i]:
                            nodes_per_layer[layer_i].append(child)
                        node_j.add_child(child)
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
    parser.add_argument("--dataset", "-d", type=str, default="phenomena/vocab/simple.json")
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

    save_path = args.dataset.split(".json")[0] + "_circuit.pkl"
    """
    with open(save_path, 'wb') as pickle_file:
        pickle.dump(circuit.to_dict(), save_path)
    """