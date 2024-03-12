
import os
from collections import defaultdict
import pickle
import json

import numpy as np
from tqdm.auto import tqdm

import datasets

pile_canonical = "/om/user/ericjm/the_pile/the_pile_test_canonical_200k"
dataset = datasets.load_from_disk(pile_canonical)

def tokenize_sample(sample):
    tokens = tokenizer(sample["text"], return_tensors='pt', 
                        max_length=1024, truncation=True)["input_ids"]
    return {"input_ids": tokens}

starting_indexes = np.array([0] + list(np.cumsum(dataset["preds_len"])))

def loss_idx_to_dataset_idx(idx):
    """given an idx in range(0, 10658635), return
    a sample index in range(0, 20000) and pred-in-sample
    index in range(0, 1023). Note token-in-sample idx is
    exactly pred-in-sample + 1"""
    sample_index = np.searchsorted(starting_indexes, idx, side="right") - 1
    pred_in_sample_index = idx - starting_indexes[sample_index]
    return int(sample_index), int(pred_in_sample_index)

def get_context(idx):
    """given idx in range(0, 10658635), return dataset sample
    and predicted token index within sample, in range(1, 1024)."""
    sample_index, pred_index = loss_idx_to_dataset_idx(idx)
    return dataset[sample_index], pred_index+1

def print_context(idx, context_length=-1):
    """
    given idx in range(0, 10658635), print prompt preceding the corresponding
    prediction, and highlight the predicted token.
    """
    sample, token_idx = get_context(idx)
    prompt = sample["split_by_token"][:token_idx]
    if context_length > 0:
        prompt = prompt[-context_length:]
    prompt = "".join(prompt)
    token = sample["split_by_token"][token_idx]
    print(prompt + "\033[41m" + token + "\033[0m")


SAVE_DIR = "/om/user/ericjm/results/dictionary-circuits/dense_clustering/exp006"

# load up idxs
with open("/om/user/ericjm/results/dictionary-circuits/dense_clustering/exp006/idxs.pkl", "rb") as f:
    idxs = pickle.load(f)

# we need to save the contexts for these idxs as a json in the following format
# we need the json to be a dictionary with idxs (converted to strings) as keys
# with the folling values: a dictionary with a key for "y", for "context", and
# for "document_idx"

# first construct this dictionary
contexts = {}
for idx in tqdm(idxs):
    document_idx, _ = loss_idx_to_dataset_idx(idx)
    document, token_idx = get_context(idx)
    tokens = document["split_by_token"]
    # prompt = tokens[:token_idx]
    # actually only include at most the last 100 tokens
    prompt = tokens[max(0, token_idx-100):token_idx]
    token = tokens[token_idx]
    contexts[str(idx)] = {"y": token, "context": prompt, "document_idx": document_idx}

# save contexts as a json
with open(os.path.join(SAVE_DIR, "contexts-pythia-70m-100k.json"), "w") as f:
    json.dump(contexts, f)

# load up clustering results
with open(os.path.join(SAVE_DIR, "clusters-pythia-70m-100k-kmeans-30k-dim.pkl"), "rb") as f:
    clusters = pickle.load(f)

clusters_json = {}
for n_clusters, result in clusters.items():
    # result is just a list of cluster assignments, no longer a tuple of lists of assignments
    results_pair = (result, None) # the streamlit expects a tuple of two lists, then ignores the second list
    clusters_json[str(n_clusters)] = list(results_pair)

# save clusters as a json
with open(os.path.join(SAVE_DIR, "clusters-pythia-70m.json"), "w") as f:
    json.dump(clusters_json, f)
