import argparse
import os
import pickle
import torch as t

from nnsight import LanguageModel
from tqdm import tqdm, trange
from collections import defaultdict
from loading_utils import (
    load_examples, load_submodule_and_dictionary, submodule_name_to_type_layer, DictionaryCfg
)
from acdc import patching_on_y, patching_on_downstream_feature
from ablation_utils import run_with_ablated_features

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
        
        self.effect_on_parents = {}

    def add_child(self, child, effect_on_parent=None):
        if "_" in self.name:
            this_layer = self.name.split("_")[0]
            this_type = self.name.split("_")[-1]
            child_layer = child.name.split("_")[0]
            child_type = child.name.split("_")[-1]
            if child_layer > this_layer:
                raise Exception(f"Invalid child: {self.name} -> {child.name}")
            elif child_layer == this_layer and "attn" == this_type: # account for the fact that attn can be child of mlp in the same transformer block
                raise Exception(f"Invalid child: {self.name} -> {child.name}")
            elif child_layer == this_layer and "mlp" == this_type and "resid" == child_type:
                raise Exception(f"Invalid child: {self.name} -> {child.name}")
            
        self.children.append(child)
        child.parents.append(self)
        child.effect_on_parents[self] = effect_on_parent

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
        self.y_threshold = 0.02
        self.feat_threshold = 0.01
        self.path_threshold = 0.01
        self.filter_proportion = 0.25

        self.root = CircuitNode("y")
        self.final_token_positions = [example["prefix_length_wo_pad"] - 1 for example in dataset]

    def _get_paths_to_root(self, lower_node, upper_node):
        for parent in upper_node.parents:
            if parent.name == "y":
                yield [lower_node, upper_node]
            else:
                for path in self._get_paths_to_root(upper_node, parent):
                    yield [lower_node] + path

    def _evaluate_effects(self, effects, final_token_positions, submodule_names, threshold, ds_node, nodes_per_submod):
        """
        Adds nodes with effect above threshold to circuit and nodes_per_submod dict.
        us: upstream
        """
        for us_submod, us_submod_name in zip(effects, submodule_names):
            us_submod_layer, us_submod_type = submodule_name_to_type_layer(us_submod_name)
            sum_indirect_effect = t.sum(effects[us_submod][final_token_positions, :], dim=0) # TODO: compute IE across all examples
            mean_indirect_effect = sum_indirect_effect / len(self.dataset)
            feats_above_threshold = (mean_indirect_effect > threshold).nonzero().flatten().tolist()
            for feature_idx in feats_above_threshold:
                us_node_name = f"{us_submod_layer}_{feature_idx}_{us_submod_type}"
                child = CircuitNode(us_node_name)
                ds_node.add_child(child, effect_on_parent=mean_indirect_effect[feature_idx].item())
                if child not in nodes_per_submod[us_submod]:
                    nodes_per_submod[us_submod].append(child)

    def locate_circuit(self, patch_method='separate'):
        num_layers = self.model.config.num_hidden_layers
        nodes_per_submod = defaultdict(list)

        # Load all submodules and dictionaries into list grouped by layer
        all_submodule_names, all_submodules, all_dictionaries = [], [], []
        for layer in range(num_layers):
            submodule_names_layer, submodules_layer, dictionaries_layer = [], [], []
            for submodule_name in self.submodules_generic:
                submodule_name = submodule_name.format(str(layer))
                submodule, dictionary = load_submodule_and_dictionary(self.model, submodule_name, self.dict_cfg)
                submodule_names_layer.append(submodule_name)
                submodules_layer.append(submodule)
                dictionaries_layer.append(dictionary)
            all_submodule_names.append(submodule_names_layer)
            all_submodules.append(submodules_layer)
            all_dictionaries.append(dictionaries_layer)

        # Iterate through layers
        for layer in trange(num_layers-1, -1, -1, desc="Layers", total=num_layers):
            submodule_names_layer, submodules_layer, dictionaries_layer = all_submodule_names[layer], all_submodules[layer], all_dictionaries[layer]
            # Effects on y (downstream)
            # Upstream: submodules of this layer
            effects_on_y = patching_on_y(self.dataset, self.model, submodules_layer, dictionaries_layer, method=patch_method).effects
            self._evaluate_effects(effects_on_y, self.final_token_positions, submodule_names_layer, self.y_threshold, self.root, nodes_per_submod)

            # Effects on submodules in this layer
            # Upstream: submodules of layers earier in the forward pass
            if layer > 0:
                for ds_submodule, ds_dictionary in zip(submodules_layer, dictionaries_layer): # iterate backwards through submodules; first submodule in all_submodules cannot be downstream
                    # Effects on downstream features
                    # Iterate backwards through submodules and measure causal effects.
                    us_submodule_names = [n for names in all_submodule_names[:layer] for n in names] # flatten all_submodule_names[:layer]
                    us_submodules = [s for submodules in all_submodules[:layer] for s in submodules]
                    us_dictionaries = [d for dictionaries in all_dictionaries[:layer] for d in dictionaries]
                    for ds_node in nodes_per_submod[ds_submodule]:
                        print(f'ds_node: {ds_node.name}')
                        ds_node_idx = int(ds_node.name.split("_")[1])
                        feat_ds_effects = patching_on_downstream_feature(
                            self.dataset,
                            self.model, 
                            us_submodules,
                            us_dictionaries,
                            ds_submodule,
                            ds_dictionary,
                            downstream_feature_id=ds_node_idx,
                            method=patch_method,
                            ).effects
                        self._evaluate_effects(feat_ds_effects, self.final_token_positions, us_submodule_names, self.feat_threshold, ds_node, nodes_per_submod)


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
        
        mean_percent_recovered = 0
        total = 0
        for example in tqdm(eval_dataset, desc="Faithfulness examples", total=len(eval_dataset)):
            with self.model.invoke(example["clean_prefix"]) as invoker:
                pass
            model_logit_diff = invoker.output.logits[:, -1, example["clean_answer"]] - \
                                invoker.output.logits[:, -1, example["patch_answer"]]

            circuit_out = run_with_ablated_features(self.model, example["clean_prefix"], self.dict_cfg.dir, self.dict_cfg.size,
                                                    feature_list, patch_vector=patch_vector, inverse=True)["model"]
            circuit_logit_diff = circuit_out.logits[:, -1, example["clean_answer"]] - \
                                    circuit_out.logits[:, -1, example["patch_answer"]]
            percent_change = (model_logit_diff - circuit_logit_diff) / model_logit_diff
            percent_recovered = 1. - percent_change
            mean_percent_recovered += percent_recovered
            total += 1
        
        mean_percent_recovered /= total
        return mean_percent_recovered.item()


    def get_feature_list(self):
        def _normalize_name(name):
            layer, feat_idx, submodule_type = name.split("_")
            return f"{submodule_type}_{layer}/{feat_idx}"

        feature_set = set()
        # Depth-first search
        def _dfs(curr_node):
            if curr_node.name != "y":
                feature_set.add(_normalize_name(curr_node.name))
            # if children, go through each
            for child in curr_node.children:
                feature_set.add(_normalize_name(child.name))
                _dfs(child)

        _dfs(self.root)
        return list(feature_set)


    def to_dict(self):
        # Depth-first search
        def _dfs(curr_node, d = {}):
            d[curr_node.name] = []
            # if children, go through each
            for child in curr_node.children:
                d[curr_node.name].append((child.name, child.effect_on_parents[curr_node]))
                _dfs(child, d=d)
            # else, return dictionary
            return d

        out_dict = _dfs(self.root)
        return out_dict


    def from_dict(self, dict_path):
        with open(dict_path, "rb") as handle:
            circuit_dict = pickle.load(handle)

        print("Loading from dictionary...")
        nodes_in_circuit = {"y": None}
        
        is_root = False
        for parent_name in circuit_dict.keys():
            if parent_name == "y":
                is_root = True
                parent_node = self.root
            else:
                if parent_name not in nodes_in_circuit:
                    parent_node = CircuitNode(parent_name)
                    nodes_in_circuit[parent_name] = parent_node
                else:
                    parent_node = nodes_in_circuit[parent_name]
            for child in circuit_dict[parent_name]:
                child_name, effect_on_parent = child
                if child_name not in nodes_in_circuit:
                    child_node = CircuitNode(child_name)
                    nodes_in_circuit[child_name] = child_node
                else:
                    child_node = nodes_in_circuit[child_name]
                parent_node.add_child(child_node, effect_on_parent=effect_on_parent)
        
        print("Circuit loaded.")


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
    parser.add_argument("--patch_method", "-p", type=str, choices=["all-folded", "separate", "ig", "exact"],
                        default="separate", help="Method to use for attribution patching.")
    parser.add_argument("--evaluate", action="store_true", help="Load and evaluate a circuit.")
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
    save_path = args.dataset.split("/")[-1].split(".json")[0] + "_circuit.pkl"

    circuit = Circuit(model, submodules, dictionary_dir, args.dictionary_size, dataset)
    if args.evaluate:
        circuit.from_dict(save_path)
        faithfulness = circuit.evaluate_faithfulness()
        print(f"Faithfulness: {faithfulness}")
    else:
        circuit.locate_circuit(patch_method=args.patch_method)
        print(circuit.to_dict())
        with open(save_path, 'wb') as pickle_file:
            pickle.dump(circuit.to_dict(), pickle_file)