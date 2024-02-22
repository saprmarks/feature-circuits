import argparse
import os
import pickle
import random
import torch as t
from graph_utils import WeightedDAG, deduce_edge_weights
from attribution import patching_effect, get_grad
from tensordict import TensorDict

from nnsight import LanguageModel
from tqdm import tqdm
from copy import deepcopy
from loading_utils import load_submodule_and_dictionary
from ablation_utils import run_with_ablated_features


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

if __name__ == "__main__":
    circuit.from_dict(save_path)
    faithfulness = circuit.evaluate_faithfulness()
    print(f"Faithfulness: {faithfulness}")
    completeness = circuit.evaluate_completeness()
    print(f"Completeness: {completeness['mean_completeness']}")
    # minimality = circuit.evaluate_minimality()
    # print(f"Minimality: {minimality['min_minimality']}")
    # print(f"Minimality per node: {minimality['minimality_per_node']}")
