"""
This script creates a database for easily accessing just the data
for an individual cluster. This should improve efficiency, since
much less data will need to be loaded into memory at once.

This script will also save everything else that's needed for the 
clustering visualization, including the mean loss curve.
"""

import os
import sys
from collections import defaultdict
import pickle
import h5py
import gzip
import glob
import io

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch as t
import torch.nn as nn
import torch.nn.functional as F

import datasets
from sklearn.cluster import SpectralClustering
from sqlitedict import SqliteDict

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

losses = np.load("/om/user/ericjm/results/trajectory-dictionaries/pythia-70m-deduped-20k/pythia-70m-deduped-timeseries.npy")
mean_loss_curve = np.mean(losses, axis=0)

# save the mean loss curve
np.save("/om/user/ericjm/results/dictionary-circuits/dense_clustering/exp008/mean_loss_curve.npy", mean_loss_curve)

with h5py.File("/om/user/ericjm/results/dictionary-circuits/dense_clustering/exp008/gradients.h5", "r") as f:
    Gs = f["gradients"][:]

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

db = SqliteDict("/om/user/ericjm/results/dictionary-circuits/dense_clustering/exp008/database.sqlite")

# let's just use the cluster size as the order for now
for clusteri in tqdm(range(4000)):

    cluster_data = dict()
    
    # index into the losses
    clusteri_idxs = [idxs[i] for i in cluster_is[new_i_to_old_i[clusteri]]]
    cluster_data['losses'] = losses[clusteri_idxs]

    # compute the cluster similarity matrix
    clusteri_Gs = Gs[cluster_is[new_i_to_old_i[clusteri]]]
    clusteri_Gs = clusteri_Gs / np.linalg.norm(clusteri_Gs, axis=1, keepdims=True)
    clusteri_C = clusteri_Gs @ clusteri_Gs.T
    clusteri_C = np.clip(clusteri_C, -1, 1)
    # compute the permutation using spectral clustering
    n_clusters = 4 if len(cluster_is[new_i_to_old_i[clusteri]]) > 4 else max(len(cluster_is[new_i_to_old_i[clusteri]]) - 1, 1)
    if n_clusters >= 2:
        sc = SpectralClustering(n_clusters=n_clusters, affinity="precomputed")
        clusteri_C_ang = 1 - np.arccos(clusteri_C) / np.pi
        clusteri_labels = sc.fit_predict(clusteri_C_ang)
        clusteri_subclusters = defaultdict(list)
        for i, c in enumerate(clusteri_labels):
            clusteri_subclusters[c].append(i)
        perm = []
        for subcluster_index in range(n_clusters):
            perm.extend(clusteri_subclusters[subcluster_index])
        permuted_C = clusteri_C[perm][:, perm]
    else:
        permuted_C = clusteri_C
    cluster_data['permuted_C'] = permuted_C

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

    cluster_data['contexts'] = contexts

    # add the cluster's circuit image, if it exists
    # these images are at /om/user/ericjm/results/dictionary-circuits/dense_clustering/exp008/circuits/plots/{clusteri}_dict10_node0.1_edge0.01_n27_aggsum.png
    # except the n27 could be a different number depending on the clusteri, so we really want to match
    # with a regex or something
    circuit_image_glob = f"/om/user/ericjm/results/dictionary-circuits/dense_clustering/exp008/circuits/plots/{clusteri}_dict10_node0.1_edge0.01_n*_aggsum.png"
    circuit_image_paths = glob.glob(circuit_image_glob)
    if len(circuit_image_paths) > 0:
        circuit_image_path = circuit_image_paths[0]
        if len(circuit_image_paths) > 1:
            print(f"Warning: multiple circuit images found for cluster {clusteri}. Using the first one.")
        with open(circuit_image_path, "rb") as f:
            circuit_image = f.read()
    else:
        circuit_image = None
    cluster_data['circuit_image'] = circuit_image

    # pickle and compress the `cluster_data`
    pickled_data = pickle.dumps(cluster_data)
    compressed_data = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed_data, mode='wb') as file:
        file.write(pickled_data)

    # Get the compressed byte string
    compressed_bytes = compressed_data.getvalue()

    # save the compressed data with sqlitedict
    db[clusteri] = compressed_bytes
    db.commit()

db.close()
