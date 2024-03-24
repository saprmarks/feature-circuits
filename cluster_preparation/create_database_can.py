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
import json
sys.path.append("/home/can/dictionary-circuits/")

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm, trange

import torch as t
import torch.nn as nn
import torch.nn.functional as F

import datasets
from sklearn.cluster import SpectralClustering
from sqlitedict import SqliteDict
from PIL import Image

######################
# Load up the dataset
######################

param_string = f"lin_effects_final-5-pos_nsamples8192_nctx64"
n_clusters = 750
n_ctx = 16
node_thresh = 0.1
save_images = True

results_dir = "/home/can/dictionary-circuits/cluster_preparation/"
dataset_dir = "/home/can/feature_clustering/app_contexts/samples8192.json"
cluster_index_dir = f"/home/can/feature_clustering/app_clusters/{param_string}.json"
losses_dir = "/home/can/data/pythia-70m-deduped-timeseries.npy"
circuit_dir = f"/share/projects/dictionary_circuits/circuits/"

# Load samples
with open(dataset_dir, "r") as f:
    samples = json.load(f)

# Load cluster indexes
with open(cluster_index_dir, "r") as f:
    cluster_mapping = json.load(f)

cluster_to_samples_map = defaultdict(list)
for sample_idx, cluster_idx in enumerate(cluster_mapping[str(n_clusters)]):
    cluster_to_samples_map[cluster_idx].append(sample_idx)



#########################
# Save the numerical data
#########################
    
with open(results_dir + "meta.json", "w") as f:
    json.dump({
    "n_clusters": n_clusters,
    "starting_cluster_idx": 0,
    "database_description": f"We show clusters based on linear effects of all SAE features on the log probability of the correct next token. We apply spectral clustering on 8125 contexts with {n_clusters} clusters in total."
}, f)


db = SqliteDict(results_dir + f"database_stats.sqlite")
for cluster_idx in trange(n_clusters, desc="Saving numerical data"):
    cluster_data = dict()

    cluster_data["cluster_idx"] = cluster_idx
    cluster_data["contexts"] = {i: samples[str(sample_idx)] for i, sample_idx in enumerate(cluster_to_samples_map[cluster_idx])}

    # Compute circuit metrics
    # Load circuit
    circuit_paths = glob.glob(circuit_dir + f"{param_string}_cluster{cluster_idx}of{n_clusters}*.pt")
    if len(circuit_paths) > 0:
        circuit_path = circuit_paths[0]
        if len(circuit_paths) > 1:
            print(f"Warning: multiple circuit images found for cluster {cluster_idx}. Using the first one.")

        cluster_data["circuit_metrics"] = dict()
        c = t.load(circuit_path)


        def feature_effects_nodes(c):
            return t.tensor([x.act.abs().sum() for n, x in c['nodes'].items() if n != 'y'], device='cuda:0')
        
        def feature_effects_writers(c):
            return t.tensor([x.act.abs().sum() for n, x in c['nodes'].items() if n[0] not in ["r", "y"]], device='cuda:0')
        
        def error_effects_nodes(c):
            return t.cat([x.resc.abs() for n, x in c['nodes'].items() if n != 'y']).to('cuda:0')

        def count_nodes(c, node_thresh=node_thresh):
            err_cnt = (error_effects_nodes(c) > node_thresh).sum().item()
            total_cnt = err_cnt
            for n, x in c['nodes'].items():
                if n == 'y':
                    continue
                total_cnt += (x.act.abs() > node_thresh).sum().item()
            return total_cnt, err_cnt
        
        def feature_effects_edges(c):
            edges = []
            for child in c['edges']:
                for parent in c['edges'][child]:
                    edges.append(c['edges'][child][parent].abs().sum())
            return t.tensor(edges, device='cuda:0')

        def interestingness_metric(c):
            feature_effects = feature_effects_nodes(c)
            err_effects = error_effects_nodes(c)
            feature_agg = (t.softmax(feature_effects, dim=0) * feature_effects).sum().item()
            err_agg = (t.softmax(err_effects, dim=0) * err_effects).sum().item()
            return feature_agg / (feature_agg + err_agg)
        
        total_nodes, error_nodes = count_nodes(c)
        cluster_data["circuit_metrics"]["n_nodes"] = total_nodes
        cluster_data["circuit_metrics"]["n_triangles"] = error_nodes
        cluster_data["circuit_metrics"]["relative_max_feature_effect_node"] = feature_effects_nodes(c).max().item() / feature_effects_nodes(c).mean().item()
        cluster_data["circuit_metrics"]["relative_max_feature_effect_edge"] = feature_effects_edges(c).max().item() / feature_effects_edges(c).mean().item()
        cluster_data["circuit_metrics"]["relative_writer_effect_node"] = feature_effects_writers(c).sum().item() / feature_effects_nodes(c).sum().item()
        cluster_data["circuit_metrics"]["relative_softmaxx_feature_effects_node"] = interestingness_metric(c)

    # pickle and compress the `cluster_data`
    pickled_data = pickle.dumps(cluster_data)
    compressed_data = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed_data, mode='wb') as file:
        file.write(pickled_data)

    # Get the compressed byte string
    compressed_bytes = compressed_data.getvalue()

    # save the compressed data with sqlitedict
    db[cluster_idx] = compressed_bytes
    db.commit()

db.close()


#######################
# Save circuit images
#######################


if save_images:
    db = SqliteDict(results_dir + f"circuit_images.sqlite")

    for cluster_idx in trange(n_clusters, desc="Saving circuit images"):
        cluster_data = dict()
        cluster_data["cluster_idx"] = cluster_idx

        # Load circuit image
        circuit_image_paths = glob.glob(circuit_dir + f"figures/{param_string}_cluster{cluster_idx}of{n_clusters}*.png")
        if len(circuit_image_paths) > 0:
            circuit_image_path = circuit_image_paths[0]
            if len(circuit_image_paths) > 1:
                print(f"Warning: multiple circuit images found for cluster {cluster_idx}. Using the first one.")
            #open circuit image path and decrease the resolution of the image while retaining size
            img = Image.open(circuit_image_path)
            scale_factor = 0.5
            new_size = (int(img.size[0] * scale_factor), int(img.size[1] * scale_factor))
            img = img.resize(new_size, Image.LANCZOS)
        else:
            img = None
        cluster_data['circuit_image'] = img

        # pickle and compress the `cluster_data`
        pickled_data = pickle.dumps(cluster_data)
        compressed_data = io.BytesIO()
        with gzip.GzipFile(fileobj=compressed_data, mode='wb') as file:
            file.write(pickled_data)

        # Get the compressed byte string
        compressed_bytes = compressed_data.getvalue()

        # save the compressed data with sqlitedict
        db[cluster_idx] = compressed_bytes
        db.commit()
    db.close()