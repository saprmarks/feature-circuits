import os
import json
from tqdm import tqdm
from collections import defaultdict

import numpy as np
import torch as t
import datasets
from transformers import AutoTokenizer

import sys
sys.path.append("/home/can/")
sys.path.append("/home/can/dictionary-circuits/")
from cluster_utils import ClusterConfig, get_string_context_y, loss_idx_to_dataset_idx
from loading_utils import submodule_name_to_type


def make_experiment_dir(ccfg, parent_dir):
    submod_type_names="-".join([submodule_name_to_type(s) for s in ['model.gpt_neox.layers.{}.mlp.dense_4h_to_h']])
    run_summary = f"clustering_{ccfg.model_name}_tloss{ccfg.loss_threshold}_ntok{ccfg.num_samples}_skip{ccfg.skip}_npos{ccfg.n_pos}_{submod_type_names}"
    results_dir = os.path.join(parent_dir, run_summary) # Create a directory for this experiment run
    os.mkdir(results_dir)
    return results_dir

def fill_sample_dict(tokenizer, ccfg, model_cache_dir, dataset, starting_indexes):
    # Load indices of tokens with low loss
    particular_model_cache_dir = os.path.join(model_cache_dir, ccfg.model_name)
    losses_cached = [f for f in os.listdir(particular_model_cache_dir) if f.endswith("losses.pt")]
    max_i = max(list(range(len(losses_cached))), key=lambda i: int(losses_cached[i].split("_")[0]))
    docs, tokens = int(losses_cached[max_i].split("_")[0]), int(losses_cached[max_i].split("_")[2])
    losses = t.load(os.path.join(particular_model_cache_dir, f"{docs}_docs_{tokens}_tokens_losses.pt"))
    c = 1 / np.log(2) # for nats to bits conversion
    token_loss_idxs = (losses < (ccfg.loss_threshold / c)).nonzero().flatten()
    token_loss_idxs = token_loss_idxs[::ccfg.skip]

    # Choose tokens with a context longer or equal to 100 tokens 
    final_token_idxs = t.zeros((ccfg.num_samples, 2)) # document index, token in doc index
    all_cnt, true_cnt = 0, 0
    progress_bar = tqdm(total=ccfg.num_samples, desc="Loading samples")

    while true_cnt < ccfg.num_samples:
        doc_index, final_token_index = loss_idx_to_dataset_idx(token_loss_idxs[all_cnt], starting_indexes)
        if final_token_index >= ccfg.n_pos:
            final_token_idxs[true_cnt] = t.tensor([doc_index, final_token_index], dtype=t.int64)
            progress_bar.update(1)
            true_cnt += 1
        all_cnt += 1
        if all_cnt == len(token_loss_idxs):
            raise ValueError(f"Not enough tokens to analyze. Loaded {true_cnt} of {ccfg.num_samples} samples.")

    # Populate dictionary with samples of (context, y)
    sample_dict = defaultdict(list)
    for j in range(ccfg.num_samples):
        document_idx, final_token_in_context_index = [int(idx) for idx in final_token_idxs[j]]
        context, y, document_idx = get_string_context_y(tokenizer, ccfg, dataset, doc_idx=document_idx, final_token_in_context_index=final_token_in_context_index)
        sample_dict[j] = dict(context=context, y=y, document_idx=document_idx)
    return sample_dict, final_token_idxs


if __name__ == "__main__":

    # Set general clustering parameters
    parent_dir = "/home/can/feature_clustering/"
    model_cache_dir = "/home/can/feature_clustering/model_cache/"
    tokenized_dataset_dir = "/home/can/data/pile_test_tokenized_600k/"
    ccfg = ClusterConfig(
        model_name="pythia-70m-deduped",
        loss_threshold=0.03,
        num_samples=int(2**14), #16k
        skip=256,
        n_pos=256,
        submodules_generic = ['model.gpt_neox.layers.{}.mlp.dense_4h_to_h'],
        dictionary_size=512*64
        )
    
    # Make dir and save the config
    results_dir = make_experiment_dir(ccfg, parent_dir)
    ccfg.results_dir = results_dir
    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(ccfg.__dict__, f)


    # Load model and full dataset
    tokenizer = AutoTokenizer.from_pretrained(
        f"EleutherAI/{ccfg.model_name}",
        revision=f"step{143000}",
        cache_dir=os.path.join(model_cache_dir, ccfg.model_name, f"step{143000}"),
    )
    dataset = datasets.load_from_disk(tokenized_dataset_dir)
    starting_indexes = np.array([0] + list(np.cumsum(dataset["preds_len"])))

    # Load and save tokens with low loss and context length longer than ccfg.n_pos
    sample_dict, final_token_idxs= fill_sample_dict(tokenizer, ccfg, model_cache_dir, dataset, starting_indexes)

    ## Save samples in string format for displaying 
    sample_dict_path = os.path.join(results_dir, "samples.json")
    with open(sample_dict_path, "w") as f:
        json.dump(sample_dict, f)
    
    ## Save final_token_idxs to torch tensor for data loading in feature cache
    # final token idx have shape (num_samples, 2) where the first column is the document index and the second column is the index of the token in the document
    final_token_idxs_path = os.path.join(results_dir, "final_token_idxs.pt")
    t.save(final_token_idxs, final_token_idxs_path)
