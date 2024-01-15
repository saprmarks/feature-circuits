import argparse
import os
import pickle
import torch as t
import regex as re

from nnsight import LanguageModel
from tqdm import tqdm
from collections import defaultdict
from dictionary_learning.buffer import ActivationBuffer
from dictionary_learning.dictionary import AutoEncoder
from loading_utils import (
    load_examples, load_submodule, submodule_type_to_name, submodule_name_to_type_layer, DictionaryCfg
)
from acdc import patching_on_y, patching_on_downstream_feature
from subnetwork import (
    Node, Subnetwork,
)
from patching import subnetwork_patch

class CircuitNode:
    def __init__(self, name, data = None, children = None, parents = None):
        self.name = name    # format: `{layer}_{idx}_{submodule_type}` OR `y`
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
        self.dict_cfg = DictionaryCfg(dictionary_dir, dictionary_size)
        self.dataset = dataset
        self.patch_token_pos = -1
        self.y_threshold = 0.05
        self.feat_threshold = 0.2
        self.path_threshold = 0.01
        self.filter_proportion = 0.25

        self.root = CircuitNode("y")

    def _get_paths_to_root(self, lower_node, upper_node):
        for parent in upper_node.parents:
            if parent.name == "y":
                yield [lower_node, upper_node]
            else:
                for path in self._get_paths_to_root(upper_node, parent):
                    yield [lower_node] + path

    def _evaluate_effects(self, effects, threshold, ds_node, nodes_per_submod):
        """
        Adds nodes with effect above threshold to circuit and nodes_per_submod dict if effect above.
        us: upstream
        """
        for us_submod_name in effects:
            us_submod_layer, us_submod_type = submodule_name_to_type_layer(us_submod_name)
            feature_indices = (effects[us_submod_name][self.patch_token_pos, :] > threshold).nonzero().flatten().tolist()
            for feature_idx in feature_indices:
                us_node_name = f"{us_submod_layer}_{feature_idx}_{us_submod_type}"
                child = CircuitNode(us_node_name)
                child.effect_on_parent = effects[us_submod_name][self.patch_token_pos, feature_idx].item()
                ds_node.add_child(child)
                if child not in nodes_per_submod[us_submod_name]:
                    nodes_per_submod[us_submod_name].append(child)
        return nodes_per_submod

    def locate_circuit(self, patch_method='separate'):
        num_layers = self.model.config.num_hidden_layers # not needed?
        nodes_per_submod = defaultdict(list)

        # List submodule names in order of forward pass
        submodule_names = []
        for layer in range(num_layers):
            for submod in self.submodules_generic: # assumes components per layer (attn, mlp, resid) are ordered by call during a forward pass
                submodule_names.append(submod.format(str(layer)))

        # Effects on y
        pat_y_out = patching_on_y(self.dataset, self.model, submodule_names, self.dict_cfg, method=patch_method)
        effects_on_y = pat_y_out.effects
        nodes_per_submod = self._evaluate_effects(effects_on_y, self.y_threshold, self.root, nodes_per_submod)

        # Effects on downstream (parent) features
        # Iterate backwards through submodules and measure causal effects.
        for ds_submod_idx, ds_submod_name in tqdm(reversed(list(enumerate(submodule_names))), desc="downstream_submodules", total=num_layers-1):
            if ds_submod_name in nodes_per_submod: # If current submodule contains relevant features
                upstream_submodule_names = submodule_names[:ds_submod_idx]
                if len(upstream_submodule_names) < 1:
                    break # current ds_submodule is the first submodule after input, no upstream_submodules left!
                for ds_node in nodes_per_submod[ds_submod_name]: # Iterate through features in current submodule
                    print(ds_node.name)
                    ds_node_idx = int(ds_node.name.split("_")[1])
                    feat_ds_effects = patching_on_downstream_feature(
                        self.dataset, 
                        self.model, 
                        upstream_submodule_names,
                        ds_submod_name,
                        downstream_feature_id=ds_node_idx,
                        dict_cfg=self.dict_cfg,
                        method=patch_method,
                        ).effects
                    nodes_per_submod = self._evaluate_effects(feat_ds_effects, self.feat_threshold, ds_node, nodes_per_submod)
        return pat_y_out
             
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
    

def run_with_ablated_features(model, example, dictionary_dir, dictionary_size, features,
                              return_submodules=None):
    """
    TODO: This method currently uses zero ablations. Should update this to also support
    naturalistic ablations.
    """
    # usage: features_per_layer[layer][submodule_type]
    features_per_layer = defaultdict(lambda: defaultdict(list))
    for feature in features:
        submodule_type, layer_and_feat_idx = feature.split("_")
        layer, feat_idx = layer_and_feat_idx.split("/")
        features_per_layer[int(layer)][submodule_type].append(int(feat_idx))

    saved_submodules = {}
    def _ablate_features(submodule_type, layer, feature_list):
        submodule_name = submodule_type_to_name(submodule_type).format(layer)
        submodule = load_submodule(model, submodule_name)
        # if there are no features to ablate, just return the submodule output
        if len(feature_list) == 0 and return_submodules and submodule_name in return_submodules:
            saved_submodules[submodule_name] = submodule.output.save()
            return
        
        # load autoencoder
        is_resid = len(submodule.output[0].shape) > 2
        if is_resid:
            submodule_width = submodule.output[0].shape[2]
        else:
            submodule_width = submodule.out_features
        autoencoder = AutoEncoder(submodule_width, dictionary_size).cuda()
        try:
            autoencoder.load_state_dict(
                t.load(os.path.join(dictionary_dir, f"{submodule_type}_out_layer{layer}/1_32768/ae.pt"))
            )
        except FileNotFoundError:
            autoencoder.load_state_dict(
                t.load(os.path.join(dictionary_dir, f"{submodule_type}_out_layer{layer}/0_32768/ae.pt"))
            )

        # encode activations into features
        if is_resid:
            x = submodule.output[0]
        else:
            x = submodule.output
        f = autoencoder.encode(x)
        for feature_idx in feature_list:
            f[:, :, feature_idx] = 0.0      # ablate features
        # replace submodule w/ autoencoder out
        if is_resid:
            submodule.output[0] = autoencoder.decode(f)
        else:
            submodule.output = autoencoder.decode(f)

        # replace activations of submodule
        if return_submodules and submodule_name in return_submodules:
            saved_submodules[submodule_name] = submodule.output.save()

    with model.invoke(example) as invoker:
        # from lowest layer to highest (does this matter in nnsight?)
        for layer in sorted(features_per_layer):
            for submodule_type in features_per_layer[layer]:
                _ablate_features(submodule_type, layer, features_per_layer[layer][submodule_type])
        if return_submodules:
            for submodule_name in return_submodules:
                if submodule_name not in saved_submodules.keys():
                    print(submodule_name)
                    submodule_type = "mlp" if "mlp" in submodule_name else "attn" if "attention" in submodule_name else "resid"
                    submodule_parts = submodule_name.split(".")
                    for idx, part in enumerate(submodule_parts):
                        if part.startswith("layer"):
                            layer = int(submodule_parts[idx+1])
                            break
                    _ablate_features(submodule_type, layer, [])
    saved_submodules["model"] = invoker.output

    return saved_submodules


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
    parser.add_argument("--patch_method", "-p", type=str, choices=["all-folded", "separate", "ig", "exact"],
                        default="separate", help="Method to use for attribution patching.")
    args = parser.parse_args()

    submodules = args.submodules
    if "," in submodules:
        submodules = submodules.split(",")
    else:
        submodules = [submodules]

    model = LanguageModel(args.model, dispatch=True)
    model.local_model.requires_grad_(True)
    dataset = load_examples(args.dataset, args.num_examples, model, pad_to_length=3)
    dictionary_dir = os.path.join(args.dictionary_dir, args.model.split("/")[-1])

    circuit = Circuit(model, submodules, dictionary_dir, args.dictionary_size, dataset)
    circuit.locate_circuit(patch_method=args.patch_method)
    print(circuit.to_dict())

    save_path = args.dataset.split("/")[-1].split(".json")[0] + "_circuit.pkl"
    with open(save_path, 'wb') as pickle_file:
        pickle.dump(circuit.to_dict(), pickle_file)