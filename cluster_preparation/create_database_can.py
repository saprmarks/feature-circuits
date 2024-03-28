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

param_string = f"lin_effects_sum-over-pos_nsamples8192_nctx64"
n_clusters = 750
n_ctx = 16
node_thresh = 0.1
save_images = False

results_dir = "/home/can/dictionary-circuits/cluster_preparation/"
dataset_dir = "/home/can/dictionary-circuits/feature_clustering/clusters/dataset_nsamples8192_nctx64_tloss0.1_filtered-induction_attn-mlp-resid_pythia-70m-deduped/samples-tloss0.1-nsamples8192-nctx64.json"
cluster_index_dir = f"/home/can/feature_clustering/app_clusters/{param_string}.json"
losses_dir = "/home/can/dictionary-circuits/cluster_preparation/pythia-70m-deduped-timeseries-can.npy"
circuit_dir = f"/share/projects/dictionary_circuits/circuits/"
similarity_dir = f"/home/can/feature_clustering/clustering_pythia-70m-deduped_tloss0.1_nsamples8192_npos64_filtered-induction_attn-mlp-resid/similarity_matrix_{param_string}.pt"
# similarity_dir = f"/home/can/feature_clustering/clustering_pythia-70m-deduped_tloss0.1_nsamples8192_npos64_filtered-induction_attn-mlp-resid/similarity_matrix_lin_effects_nsamples8192_sum-over-pos_nctx64.pt"

# Load samples
with open(dataset_dir, "r") as f:
    samples = json.load(f)

# Load cluster indexes
with open(cluster_index_dir, "r") as f:
    cluster_mapping = json.load(f)

cluster_to_samples_map = defaultdict(list)
for sample_idx, cluster_idx in enumerate(cluster_mapping[str(n_clusters)]):
    cluster_to_samples_map[cluster_idx].append(sample_idx)

# Load losses
losses = np.load(losses_dir)
mean_loss_curve = np.mean(losses, axis=0)
# save the mean loss curve
np.save("mean_loss_curve.npy", mean_loss_curve)

# Load similarity matrix
full_similarity_matrix = t.load(similarity_dir)
full_similarity_matrix = full_similarity_matrix.cpu().numpy()

######################
    # Metric functions
######################

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


#########################
# Save the numerical data
#########################
    
with open(results_dir + "meta.json", "w") as f:
    json.dump({
    "n_clusters": n_clusters,
    "starting_cluster_idx": 0,
    "database_description": f"We show clusters based on linear effects of all SAE features on the log probability of the correct next token. We apply spectral clustering on 8125 contexts with {n_clusters} clusters in total."
}, f)


db = SqliteDict(results_dir + f"database.sqlite")
metric_dict = defaultdict(list)
for cluster_idx in trange(n_clusters, desc="Saving numerical data"):
    # Cluster stats
    cluster_data = dict()
    cluster_data["cluster_idx"] = cluster_idx
    cluster_data["contexts"] = {i: samples[str(sample_idx)] for i, sample_idx in enumerate(cluster_to_samples_map[cluster_idx])}
    cluster_data['losses'] = losses[cluster_to_samples_map[cluster_idx]]
    cluster_data["circuit_metrics"] = dict()

    # Compute circuit metrics, knowledge of all samples is requires to sort
    # Load circuit
    circuit_paths = glob.glob(circuit_dir + f"{param_string}_cluster{cluster_idx}of750_*.pt")
    if len(circuit_paths) > 0:
        circuit_path = circuit_paths[0]
        if len(circuit_paths) > 1:
            print(f"Warning: multiple circuit images found for cluster {cluster_idx}. Using the first one.")
        c = t.load(circuit_path)
        
        cluster_data['circuit_metrics']['n_samples'] = len(cluster_data["contexts"])
        total_nodes, error_nodes = count_nodes(c)
        cluster_data['circuit_metrics']["n_nodes"] = total_nodes
        cluster_data['circuit_metrics']["n_triangles"] = error_nodes
        cluster_data['circuit_metrics']["relative_max_feature_effect_node"] = feature_effects_nodes(c).max().item() / feature_effects_nodes(c).mean().item()
        cluster_data['circuit_metrics']["relative_max_feature_effect_edge"] = feature_effects_edges(c).max().item() / feature_effects_edges(c).mean().item()
        cluster_data['circuit_metrics']["relative_writer_effect_node"] = feature_effects_writers(c).sum().item() / feature_effects_nodes(c).sum().item()
        cluster_data['circuit_metrics']["relative_softmaxx_feature_effects_node"] = interestingness_metric(c)

        metric_dict["n_samples"].append(cluster_data['circuit_metrics']['n_samples'])
        metric_dict["n_nodes"].append(cluster_data['circuit_metrics']['n_nodes'])
        metric_dict["n_triangles"].append(cluster_data['circuit_metrics']['n_triangles'])
        metric_dict["relative_max_feature_effect_node"].append(cluster_data['circuit_metrics']['relative_max_feature_effect_node'])
        metric_dict["relative_max_feature_effect_edge"].append(cluster_data['circuit_metrics']['relative_max_feature_effect_edge'])
        metric_dict["relative_writer_effect_node"].append(cluster_data['circuit_metrics']['relative_writer_effect_node'])
        metric_dict["relative_softmaxx_feature_effects_node"].append(cluster_data['circuit_metrics']['relative_softmaxx_feature_effects_node'])

    # Compute similarity matrix
    sample_idxs = cluster_to_samples_map[cluster_idx]
    C = full_similarity_matrix[sample_idxs][:, sample_idxs]

    clusteri_C = np.clip(C, -1, 1)
    # compute the permutation using spectral clustering
    n_clusters = 4 if len(sample_idxs) > 4 else max(len(sample_idxs) - 1, 1)
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

# Sort and save metric dict
for key in metric_dict:
    metric_dict[key] = np.argsort(metric_dict[key])[::-1].tolist()

with open(results_dir + "metrics.json", "w") as f:
    json.dump(metric_dict, f)


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


# #%%
# import numpy as np
# import torch as t
# losses_eric = np.load("/home/can/dictionary-circuits/cluster_preparation/pythia-70m-deduped-timeseries-can.npy")
# losses_can = t.load('/home/can/feature_clustering/model_cache/pythia-70m-deduped/180000_docs_93277607_tokens_losses.pt')
# # %%
# j = 3100
# for i in range(j, j+100):
#     print(i, losses_eric[i, -1], losses_can[i])
# # %%
    
# losses_can.shape
# # %%
# losses_eric.shape
# # %%


# # Load similarity matrix
# import torch as t
# import matplotlib.pyplot as plt
# import numpy as np
# random_integers = np.random.randint(0, 8192, size=100)

# similarity_dir = f"/home/can/feature_clustering/clustering_pythia-70m-deduped_tloss0.1_nsamples8192_npos64_filtered-induction_attn-mlp-resid/similarity_matrix_lin_effects_sum-over-pos_nsamples8192_nctx64.pt"
# full_similarity_matrix = t.load(similarity_dir)
# full_similarity_matrix = full_similarity_matrix.cpu().numpy()[random_integers][:, random_integers]

# plt.imshow(full_similarity_matrix, cmap='rainbow')
# plt.colorbar()

# # %%
