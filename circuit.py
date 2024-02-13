import argparse
import os
import pickle
import random
import torch as t

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
    def __init__(self, name, accumulated_gradient = None, data = None, children = None, parents = None):
        self.name = name    # format: `{layer}_{idx}_{submodule_type}` OR `y`
        self.data = data    # TODO: 10 sentences w/ activated tokens?
        self.accumulated_gradient = accumulated_gradient # Accumulated gradient of y w.r.t. this node via paths in the circuit
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
    def __init__(self, model, submodules, dictionary_dir, dictionary_size, dataset,
                 threshold=0.05, ):
        self.model = model
        self.submodules_generic = submodules
        self.dict_cfg = DictionaryCfg(dictionary_dir, dictionary_size)
        self.dataset = dataset
        self.patch_token_pos = -1
        self.y_threshold = threshold
        self.feat_threshold = threshold
        self.path_threshold = threshold

        self.root = CircuitNode("y", accumulated_gradient=1) # (d/dy)y = 1

    def _get_paths_to_root(self, lower_node, upper_node):
        for parent in upper_node.parents:
            if parent.name == "y":
                yield [lower_node, upper_node]
            else:
                for path in self._get_paths_to_root(upper_node, parent):
                    yield [lower_node] + path
    
    
    def _normalize_name(self, name):
        layer, feat_idx, submodule_type = name.split("_")
        return f"{submodule_type}_{layer}/{feat_idx}"


    def _evaluate_effects(self, effects, submodule_names, threshold, ds_node, nodes_per_submod):
        """
        Adds nodes with effect above threshold to circuit and nodes_per_submod dict.
        us: upstream
        """
        for us_submod, us_submod_name in zip(effects, submodule_names):
            us_submod_layer, us_submod_type = submodule_name_to_type_layer(us_submod_name)
            feats_above_threshold = (effects[us_submod] > threshold).nonzero().flatten().tolist()
            for feature_idx in feats_above_threshold:
                us_node_name = f"{us_submod_layer}_{feature_idx}_{us_submod_type}"
                child = CircuitNode(us_node_name, accumulated_gradient=None)
                ds_node.add_child(child, effect_on_parent=effects[us_submod][feature_idx].item())
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
            effects_on_y, _ = patching_on_y(self.model, self.dataset, submodules_layer, dictionaries_layer, method=patch_method)
            effects_on_y = effects_on_y.effects
            self._evaluate_effects(effects_on_y, submodule_names_layer, self.y_threshold, self.root, nodes_per_submod)

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
                        # print(f'ds_node: {ds_node.name}')
                        ds_node_idx = int(ds_node.name.split("_")[1])
                        feat_ds_effects, _ = patching_on_downstream_feature(
                            self.model, 
                            self.dataset,
                            us_submodules,
                            us_dictionaries,
                            ds_submodule,
                            ds_dictionary,
                            downstream_feature_id=ds_node_idx,
                            method=patch_method,
                            )
                        feat_ds_effects = feat_ds_effects.effects
                        self._evaluate_effects(feat_ds_effects, us_submodule_names, self.feat_threshold, ds_node, nodes_per_submod)


    def _evaluate_effects_chainrule(self, effects, submodule_names, threshold, ds_node, nodes_per_submod, grads_y_wrt_us_features):
        """
        Adds nodes with effect above threshold to circuit and nodes_per_submod dict.
        us: upstream, the nodes where outputs are patched
        """
        for us_submod, us_submod_name in zip(effects, submodule_names):
            us_submod_layer, us_submod_type = submodule_name_to_type_layer(us_submod_name)
            feats_above_threshold = (effects[us_submod] > threshold).nonzero().flatten().tolist()
            for feature_idx in feats_above_threshold:
                us_node_name = f"{us_submod_layer}_{feature_idx}_{us_submod_type}"
                this_node = CircuitNode(name=us_node_name, accumulated_gradient=grads_y_wrt_us_features[feature_idx])

                # check whether node already exists in circuit
                if this_node in nodes_per_submod[us_submod]:
                    for node in nodes_per_submod[us_submod]:    # TODO: make more efficient. can't index sets (use dict?)
                        if node.name == this_node.name:
                            us_node = node
                            break
                    # us_node = nodes_per_submod[us_submod][this_node.name]
                    us_node.accumulated_gradient += grads_y_wrt_us_features[feature_idx]
                    
                else:
                    us_node = this_node
                    nodes_per_submod[us_submod].add(us_node)
                ds_node.add_child(us_node, effect_on_parent=effects[us_submod][feature_idx].item())


    def locate_circuit_chainrule(self, patch_method='separate', sequence_aggregation='final_pos_only'):
        num_layers = self.model.config.num_hidden_layers
        nodes_per_submod = defaultdict(set)

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
            effects_on_y, grads_y_wrt_us_features = patching_on_y(
                self.model, 
                self.dataset, 
                submodules_layer, 
                dictionaries_layer, 
                method=patch_method, 
                grad_y_wrt_downstream=1, 
                sequence_aggregation=sequence_aggregation
                ) # (d/dy)y = 1
            effects_on_y = effects_on_y.effects
            self._evaluate_effects_chainrule(effects_on_y, submodule_names_layer, self.y_threshold, self.root, nodes_per_submod, grads_y_wrt_us_features)

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
                        # print(f'ds_node: {ds_node.name}')
                        ds_node_idx = int(ds_node.name.split("_")[1])
                        feat_ds_effects, grads_y_wrt_us_features = patching_on_downstream_feature(
                            self.model, 
                            self.dataset,
                            us_submodules,
                            us_dictionaries,
                            ds_submodule,
                            ds_dictionary,
                            downstream_feature_id=ds_node_idx,
                            grad_y_wrt_downstream=ds_node.accumulated_gradient,
                            method=patch_method,
                            sequence_aggregation=sequence_aggregation,
                            )
                        feat_ds_effects = feat_ds_effects.effects
                        self._evaluate_effects_chainrule(feat_ds_effects, us_submodule_names, self.feat_threshold, ds_node, nodes_per_submod, grads_y_wrt_us_features)


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

    def get_feature_list(self):
        feature_set = set()
        # Depth-first search
        def _dfs(curr_node):
            if curr_node.name != "y":
                feature_set.add(self._normalize_name(curr_node.name))
            # if children, go through each
            for child in curr_node.children:
                feature_set.add(self._normalize_name(child.name))
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