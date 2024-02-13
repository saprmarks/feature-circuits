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
from cluster_utils import ClusterConfig, get_tokenized_context_y, get_string_context_y, loss_idx_to_dataset_idx
from loading_utils import submodule_name_to_type


def make_experiment_dir(ccfg, parent_dir):
    submod_type_names="-".join([submodule_name_to_type(s) for s in ccfg.submodules_generic])
    run_summary = f"clustering_{ccfg.model_name}_tloss{ccfg.loss_threshold}_nsamples{ccfg.num_samples}_npos{ccfg.n_pos}_filtered-induction_{submod_type_names}"
    results_dir = os.path.join(parent_dir, run_summary) # Create a directory for this experiment run
    os.mkdir(results_dir)
    return results_dir

def contains_skip_trigram(context_ids, answer_id):
    """
    Check whether the token is a skip trigram
    context: list of token_ids
    """
    final_token_id = context_ids[-1]
    final_token_occurences = t.where(context_ids == final_token_id)[0] # Find occurences of final token in string
    potential_answer_idxs = (final_token_occurences + 1)[:-1] # It's a skip trigram if the token is the answer to the previous token
    if t.any(context_ids[potential_answer_idxs] == answer_id):
        return True
    return False

def fill_sample_dict(tokenizer, ccfg, losses_dir, dataset, starting_indexes, n_docs=600000):
    # Load indices of tokens with low loss
    losses = t.load(losses_dir) # Loss for predicting the next token
    token_loss_idxs = (losses < ccfg.loss_threshold).nonzero().flatten() # Indices of final tokens in context with low loss on the next token prediction

    # Find final tokens with 
    # 1. loss lower than ccfg.loss_threshold
    # 2. a context longer or equal to ccfg.n_pos
    # 3. not containing skip trigrams
    valid_final_token_idxs = t.zeros((ccfg.num_samples, 2)) # document index, token in doc index
    valid_cnt = 0
    sample_dict = defaultdict(list) # Populate dictionary with samples of (context, y)
    progress_bar = tqdm(total=ccfg.num_samples, desc="Loading samples")

    for doc_idx in range(n_docs):
        if valid_cnt >= ccfg.num_samples:
            break
        min_starting_index = starting_indexes[doc_idx] + ccfg.n_pos # Maintain minimum context length
        doc_token_loss_idxs = token_loss_idxs[(token_loss_idxs >= min_starting_index) & (token_loss_idxs < starting_indexes[doc_idx+1])]
        doc_token_loss_idxs -= starting_indexes[doc_idx]

        for final_token_idx in doc_token_loss_idxs:
            context_ids, answer_id, doc_idx = get_tokenized_context_y(ccfg, dataset, doc_idx=doc_idx, final_token_in_context_index=final_token_idx)
            context_ids = t.tensor(context_ids, dtype=t.int64)
            if not contains_skip_trigram(context_ids, answer_id):
                valid_final_token_idxs[valid_cnt] = t.tensor([doc_idx, final_token_idx], dtype=t.int64)
                sample_dict[valid_cnt] = dict(
                    context=tokenizer.decode(context_ids), 
                    answer=tokenizer.decode(answer_id), 
                    document_idx=doc_idx
                    )
                valid_cnt += 1
                progress_bar.update(1)
                break # we only collect one sample per document
                
    if valid_cnt < ccfg.num_samples:
        raise ValueError(f"Not enough tokens to analyze. Loaded {valid_cnt} valid samples of {ccfg.num_samples} required samples.")
    return sample_dict, valid_final_token_idxs


if __name__ == "__main__":

    # Set general clustering parameters
    parent_dir = "/home/can/feature_clustering/"
    model_cache_dir = "/home/can/feature_clustering/model_cache/"
    losses_dir = "/home/can/feature_clustering/model_cache/pythia-70m-deduped/180000_docs_93277607_tokens_losses.pt"
    tokenized_dataset_dir = "/home/can/data/pile_test_tokenized_600k/"
    ccfg = ClusterConfig(
        model_name="pythia-70m-deduped",
        loss_threshold=0.1,
        num_samples=2**13, # 8192
        n_pos=64,
        submodules_generic = ['model.gpt_neox.layers.{}.mlp.dense_4h_to_h', 'model.gpt_neox.layers.{}.attention.dense', 'model.gpt_neox.layers.{}'],
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
    sample_dict, final_token_idxs= fill_sample_dict(tokenizer, ccfg, losses_dir, dataset, starting_indexes, n_docs=600000)

    ## Save samples in string format for displaying 
    sample_dict_path = os.path.join(results_dir, "samples.json")
    with open(sample_dict_path, "w") as f:
        json.dump(sample_dict, f)
    
    ## Save final_token_idxs to torch tensor for data loading in feature cache
    # final token idx have shape (num_samples, 2) where the first column is the document index and the second column is the index of the token in the document
    final_token_idxs_path = os.path.join(results_dir, "final_token_idxs.pt")
    t.save(final_token_idxs, final_token_idxs_path)
