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


db = SqliteDict("/om/user/ericjm/results/dictionary-circuits/dense_clustering/exp008/database.sqlite")

# let's just use the cluster size as the order for now
for clusteri in tqdm(range(4000)):

    # load up this cluster's data
    compressed_bytes = db[clusteri]
    decompressed_object = io.BytesIO(compressed_bytes)
    with gzip.GzipFile(fileobj=decompressed_object, mode='rb') as file:
        cluster_data = pickle.load(file)

    # add the cluster's circuit image, if it exists
    # these images are at /om/user/ericjm/results/dictionary-circuits/dense_clustering/exp008/circuits/plots/{clusteri}_dict10_node0.1_edge0.01_n27_aggsum.png
    # except the n27 could be a different number depending on the clusteri, so we really want to match
    # with a regex or something
    # circuit_image_glob = f"/om/user/ericjm/results/dictionary-circuits/dense_clustering/exp008/circuits/plots/{clusteri}_dict10_node0.1_edge0.01_n*_aggsum.png"
    # circuit_image_paths = glob.glob(circuit_image_glob)
    # if len(circuit_image_paths) > 0:
    #     circuit_image_path = circuit_image_paths[0]
    #     if len(circuit_image_paths) > 1:
    #         print(f"Warning: multiple circuit images found for cluster {clusteri}. Using the first one.")
    #     with open(circuit_image_path, "rb") as f:
    #         circuit_image = f.read()
    # else:
    #     circuit_image = None
    # cluster_data['circuit_image'] = circuit_image

    # add the cluster's graphviz .dot string, if it exists
    # these are at /om/user/ericjm/results/dictionary-circuits/dense_clustering/exp008/circuits/graphviz_dots/{clusteri}_dict10_node0.1_edge0.01_n27_aggsum.dot 
    # except the n27 could be a different number depending on the clusteri, so we really want to match
    # with a regex or something
    dot_glob = f"/om/user/ericjm/results/dictionary-circuits/dense_clustering/exp008/circuits/graphviz_dots/{clusteri}_dict10_node0.1_edge0.01_n*_aggsum.dot"
    dot_paths = glob.glob(dot_glob)
    if len(dot_paths) > 0:
        dot_path = dot_paths[0]
        if len(dot_paths) > 1:
            print(f"Warning: multiple .dot files found for cluster {clusteri}. Using the first one.")
        with open(dot_path, "r") as f:
            dot_string = f.read()
    else:
        dot_string = None
    cluster_data['graphviz_dot'] = dot_string

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
