import argparse
import os
import torch as t

from nnsight import LanguageModel
from tqdm import tqdm
from collections import defaultdict
from dictionary_learning.buffer import ActivationBuffer
from dictionary_learning.dictionary import AutoEncoder
from causal_search import (
    load_examples, load_submodule, compare_probs,
    attribution_patching, attribution_patching_wrt_features
)


class CircuitNode:
    def __init__(self, name, data = None, children = None):
        self.name = name    # format: `{layer}_{idx}` OR `y`
        self.data = data    # TODO: 10 sentences w/ activated tokens?
        if not children:
            self.children = []

    def add_child(self, child):
        if "_" in self.name:
            this_layer = self.name.split("_")[0]
            child_layer = child.name.split("_")[0]
            if child_layer >= this_layer:
                raise Exception(f"Invalid child: {self.name} -> {child.name}")
        self.children.append(child)

    def __eq__(self, other):
        return self.name == other.name
    
    def __hash__(self):
        return hash(self.name)


class Circuit:
    def __init__(self, model, submodule, dictionary_dir, dictionary_size,
                 module_type="mlp", metric_fn = compare_probs):
        self.model = model
        self.submodule_generic = submodule
        self.dictionary_dir = dictionary_dir
        self.dictionary_size = dictionary_size
        self.module_type = module_type
        self.metric_fn = metric_fn
        self.y_threshold = 0.05
        self.feat_threshold = 0.01

        self.root = CircuitNode("y")
    
    def load_dictionary(self, layer, submodule):
        dict_path = os.path.join(self.dictionary_dir,
                                 f"{self.module_type}_layer_{layer}",
                                 f"0_{self.dictionary_size}/ae.pt")
        submodule_width = submodule.out_features
        autoencoder = AutoEncoder(submodule_width, self.dictionary_size).cuda()
        # TODO: add support for both of these cases to the `load_state_dict` method
        try:
            autoencoder.load_state_dict(t.load(args.autoencoder))
        except TypeError:
            autoencoder.load_state_dict(t.load(args.autoencoder).state_dict())
        return autoencoder

    # TODO: test   
    def locate_circuit(self):
        num_layers = self.model.config["num_hidden_layers"]
        nodes_per_layer = defaultdict(list)
        # Iterate backwards through layers. Establish causal effects
        # TODO: change upper limit to num_layers-1
        for layer_i in tqdm(range(num_layers-2, -1, -1), desc="Layer",
                            total=num_layers):
            # First, get effect on output y
            submodule_i_name = self.submodule_generic.format(str(layer_i))
            submodule_i = load_submodule(self.model, submodule_i_name)
            dictionary_i = self.load_dictionary(layer_i, submodule_i)
            effect_on_y = attribution_patching(self.model, submodule_i, dictionary_i,
                                               self.dataset, metric_fn=self.metric_fn)
            # if effect greater than threshold, add to graph
            indices = (effect_on_y > self.y_threshold).nonzero().flatten().tolist()
            for index in indices:
                node_name = f"{layer_i}_{index}"
                child = CircuitNode(node_name)
                nodes_per_layer[layer_i].append(child)
                self.root.add_child(child)

            # Second, get effect on other features already in the graph (above this feature)
            # TODO: change upper limit to num_layers-1
            for layer_j in tqdm(range(num_layers-2, layer_i, -1), leave=False, desc="Layers above",
                                total = num_layers - layer_i):
                # TODO: causal effect of lower features i on upper features j
                submodule_j_name = self.submodule_generic.format(str(layer_j))
                submodule_j = load_submodule(self.model, submodule_j_name)
                dictionary_j = self.load_dictionary(layer_j, submodule_j)
                for node_j in nodes_per_layer[layer_j]:
                    feat_idx_j = node_j.name.split("_")[1]
                    effect_on_feat_j = attribution_patching_wrt_features(self.model, submodule_i, submodule_j,
                                            dictionary_i, dictionary_j, feat_idx_j, self.dataset)
                    indices = (effect_on_feat_j > self.feat_threshold).nonzero().flatten().tolist()
                    for index in indices:
                        node_name = f"{layer_i}_{index}"
                        child = CircuitNode(node_name)
                        if child not in nodes_per_layer[layer_i]:
                            nodes_per_layer[layer_i].append(child)
                        node_j.add_child(child)

    def to_dict(self):
        # Depth-first search
        def _dfs(curr_node, out_dict = {}):
            out_dict[curr_node.name] = []
            # if children, go through each
            for child in curr_node.children:
                out_dict[curr_node.name].append(child.name)
                return _dfs(child, out_dict)
            # else, return dictionary
            return out_dict

        out_dict = _dfs(self.root)
        return out_dict
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="EleutherAI/pythia-70m-deduped"
                        help="Name of model on which dictionaries were trained.")
    parser.add_argument("--submodule", "-s", type=str, default="model.gpt_neox.layers.{}.mlp.dense_4h_to_h",
                        help="Name of submodule on which dictionaries were trained (with `{}` where the layer number should be).")
    parser.add_argument("--dictionary_dir", "-d", type=str,
                        default="autoencoders/ae_mlp3_c4_lr0.0001_resample25000_dict32768.pt")
    parser.add_argument("--dictionary_size", "-S", type=int, default=32768,
                        default="Width of trained dictionaries.")
    parser.add_argument("--dataset", "-d", type=str,
                        default="phenomena/vocab/simple.json")
    parser.add_argument("--num_examples", "-n", type=int, default=100,
                        help="Number of example pairs to use in the causal search.")
    parser.add_argument("--metric_name", "-f", type=str, default="compare_probs",
                        help="Method for determining causal effect of feature on output.")
    args = parser.parse_args()

    metric_name_to_fn = {
        "compare_probs": compare_probs
    }

    model = LanguageModel(args.model, dispatch=True)
    model.local_model.requires_grad_(True)
    dataset = load_examples(args.dataset, args.num_examples, model)

    metric_fn = metric_name_to_fn[args.metric_name]
    circuit = Circuit(args.model, args.submodule, args.dictionary_dir, args.dictionary_size,
                      dataset, metric_fn=metric_fn)
    circuit.locate_circuit()