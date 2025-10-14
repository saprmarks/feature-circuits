import os
import sys
from collections import defaultdict
import pickle
import h5py
import json

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch as t
import torch.nn as nn
import torch.nn.functional as F

import datasets

sys.path.append("/om2/user/ericjm/dictionary-circuits")
from circuit_triangles import get_circuit_cluster

######################
# Load up the dataset
######################
pile_canonical = "/om/user/ericjm/the_pile/the_pile_test_canonical_200k"
dataset = datasets.load_from_disk(pile_canonical)

starting_indexes = np.array([0] + list(np.cumsum(dataset["preds_len"])))

def loss_idx_to_dataset_idx(idx):
    """given an idx, return a document index and pred-in-sample
    index in range(0, 1023). Note token-in-sample idx is
    exactly pred-in-sample + 1. So the pred_in_sample_index is the index
    into the sequence above will the model will genenerate a prediction for the
    token at the pred_in_sample_index + 1."""
    sample_index = np.searchsorted(starting_indexes, idx, side="right") - 1
    pred_in_sample_index = idx - starting_indexes[sample_index]
    return int(sample_index), int(pred_in_sample_index)

def get_context(idx):
    """given idx, return dataset document and the index of the token 
    corresponding to the given idx within that document, in range(1, 1024)."""
    sample_index, pred_index = loss_idx_to_dataset_idx(idx)
    return dataset[sample_index], pred_index+1

def print_context(idx, context_length=-1):
    """
    given idx, print the context preceding the corresponding
    token as well as the token itself, and highlight the token.
    """
    sample, token_idx = get_context(idx)
    prompt = sample["split_by_token"][:token_idx]
    if context_length > 0:
        prompt = prompt[-context_length:]
    prompt = "".join(prompt)
    token = sample["split_by_token"][token_idx]
    print(prompt + "\033[41m" + token + "\033[0m")

clusteris_range = list(range(800, 1000))

if __name__ == '__main__':
    # get arg (integer)
    argi = int(sys.argv[1])
    clusteri = clusteris_range[argi]
    print(f"Running on cluster {clusteri}")

    with open("/om/user/ericjm/results/dictionary-circuits/dense_clustering/exp008/idxs.pkl", "rb") as f:
        idxs = pickle.load(f)

    with open(os.path.join("/om/user/ericjm/results/dictionary-circuits/dense_clustering/exp008/clusters-pythia-70m-deduped-100k-kmeans-30k-dim.pkl"), "rb") as f:
        clusters = pickle.load(f)

    clusters = clusters[4000]
    cluster_is = defaultdict(list)
    for i, c in enumerate(clusters):
        cluster_is[c].append(i)
    # create a new cluster label (indexing) scheme so that the largest cluster is cluster 0
    # and the second largest is cluster 1, etc.
    new_i_to_old_i = np.argsort([len(cluster_is[c]) for c in range(4000)])[::-1]

    clusteri_idxs = [idxs[i] for i in cluster_is[new_i_to_old_i[clusteri]]]

    # compute the contexts
    contexts = {}
    for idx in clusteri_idxs:
        document_idx, _ = loss_idx_to_dataset_idx(idx)
        document, token_idx = get_context(idx)
        tokens = document["split_by_token"]
        # prompt = tokens[:token_idx]
        # actually only include at most the last 100 tokens
        prompt = tokens[max(0, token_idx-100):token_idx]
        token = tokens[token_idx]
        contexts[idx.item()] = {"answer": token, "context": prompt, "document_idx": document_idx}
    [contexts[k]['answer'] for k in contexts.keys()]

    get_circuit_cluster(
        dataset=contexts,
        dict_path="/om/user/ericjm/dictionary-circuits/pythia-70m-deduped/",
        batch_size=5,
        dataset_name=str(clusteri),
        circuit_dir="/om/user/ericjm/results/dictionary-circuits/dense_clustering/exp008/circuits",
        plot_dir="/om/user/ericjm/results/dictionary-circuits/dense_clustering/exp008/circuits/plots",
    )
