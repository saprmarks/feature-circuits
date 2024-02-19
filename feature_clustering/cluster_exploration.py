#%%
# Imports and environment setup

import json
import os
from tqdm import tqdm, trange
import numpy as np
import torch as t
import torch.nn.functional as F
from collections import defaultdict
from matplotlib import pyplot as plt
from cluster_utils import ClusterConfig, get_pos_aggregation_description, pattern_matrix_pos_aggregated
device = "cuda:0"

#%%
# Helper functions

def compute_similarity_matrix(X, angular=False):
    X_norm = F.normalize(X, p=2, dim=-1)
    C = t.matmul(X_norm, X_norm.T) # cos sim
    if angular:
        C = t.clamp(C, -1, 1)
        C = 1 - t.acos(C) / np.pi
    return C

def compute_similarity_matrix_between_clusters(X, cluster_1_idxs, cluster_2_idxs, angular=False):
    X_norm = F.normalize(X, p=2, dim=-1)
    C = t.matmul(X_norm[cluster_1_idxs], X_norm[cluster_2_idxs].T) # cos sim
    if angular:
        C = t.clamp(C, -1, 1)
        C = 1 - t.acos(C) / np.pi
    return C

def find_pairwise_distances(X, angular=False):
    C = compute_similarity_matrix(X, angular=angular)
    # Only select values in the upper triangle
    distances = []
    for row in range(C.shape[0]):
        for col in range(row+1, C.shape[1]):
            distances.append(C[row, col].item())
    return t.tensor(distances)

def select_upper_triangle(C):
    # Only select values in the upper triangle
    print(f'selecting upper triangle of shape {C.shape}')
    distances = []
    for row in range(C.shape[0]):
        for col in range(row+1, C.shape[1]):
            distances.append(C[row, col].item())
    print(f'upper triangle has {len(distances)} elements')
    return t.tensor(distances)

def find_centroids(X, clusters, n_clusters):
    centroid_idxs = t.zeros(n_clusters, dtype=t.int)
    for cluster_idx in range(n_clusters):
        cluster_mask = t.tensor(clusters[str(n_clusters)]) == cluster_idx
        cluster = X[cluster_mask]
        cluster_mean = cluster.mean(dim=0)
        distances = t.norm(cluster - cluster_mean, dim=1)
        centroid_idx = t.argsort(distances)[0]
        centroid_idxs[cluster_idx] = centroid_idx
    return centroid_idxs




#%%
########## Analysis of Clusters ##########

data_source = "lin_effects" # activations or lin_effects
pos_aggregation = "sum" # "sum" or int x for final x positions
nsamples = 1024
nctx = 32
is_dense = True

if is_dense:
    dense_name = "dense_"
else:
    dense_name = ""

results_dir = f"/home/can/feature_clustering/clustering_pythia-70m-deduped_tloss0.1_nsamples{nsamples}_npos{nctx}_filtered-induction_mlp-attn-resid"
ccfg = ClusterConfig(**json.load(open(os.path.join(results_dir, "config.json"), "r")))
X = t.load(os.path.join(results_dir, dense_name + data_source + ".pt"))
X, aggregation_name = pattern_matrix_pos_aggregated(X, ccfg, pos_aggregation)
X = X.to_dense()
aggregation_name = get_pos_aggregation_description(pos_aggregation)
cluster_filename = dense_name + f"clusters_{data_source}_nsamples{ccfg.num_samples}_nctx{ccfg.n_pos}_{aggregation_name}.json"
activation_clusters = json.load(open(os.path.join(results_dir, cluster_filename), "r"))
#load similarity matrix
C = t.load(os.path.join(results_dir, dense_name + f"similarity_matrix_{data_source}_nsamples{ccfg.num_samples}_nctx{ccfg.n_pos}_{aggregation_name}.pt"))
C = C.cpu()

#%%
# Inspect single cluster

n_clusters = 500
cluster_idx = 327

act_clusters = t.tensor(activation_clusters[str(n_clusters)])
cluster_indices = t.where(act_clusters == cluster_idx)[0]
pairwise_distances = find_pairwise_distances(X[cluster_indices], angular=False)

print(f'cluster {n_clusters}/{cluster_idx} has {cluster_indices.shape[0]} samples')
# print(f'unique distances {pairwise_distances.unique()}\n\n\n')


# Print contexts in clusters
samples = json.load(open(os.path.join(results_dir, "samples.json"), "r"))
samples_in_cluster = defaultdict(list)
for idx in cluster_indices:
    idx = str(idx.item())
    print(f'index: {idx}')
    print(f'context: {samples[idx]["context"]}')
    print(f'answer: {samples[idx]["answer"]}')
    print('---\n\n')
    samples_in_cluster[idx] = samples[idx]

# save samples
# json.dump(samples_in_cluster, open(os.path.join(results_dir, f"contexts_lin-effects_final1pos_nclusters{n_clusters}_clusteridx{cluster_idx}.json"), "w"), indent=4)


#%%
# Load intra-cluster distances for all n_clusters

# Compute C
# select rows, cols of C that are in the cluster

aggregated_stats = defaultdict(list)
intra_cluster_distances = defaultdict(lambda: t.zeros(0))
for n_clusters in tqdm(ccfg.cluster_counts):
    for cluster_idx in range(n_clusters):
        cluster_mask = t.tensor(activation_clusters[str(n_clusters)]) == cluster_idx
        n_elements = cluster_mask.sum()
        if n_elements > 1:
            C_cluster = C[cluster_mask][:, cluster_mask]
            C_upper_triag = t.gather(C_cluster, 0, t.triu(t.ones(n_elements, n_elements, dtype=t.long), 1)).flatten()
            aggregated_stats[n_clusters].append([C_upper_triag.mean().item(), cluster_idx, n_elements.item()])
            intra_cluster_distances[n_clusters] = t.hstack((intra_cluster_distances[n_clusters], C_upper_triag))
print(f'columns: mean distance, cluster_idx, n_samples in cluster')
aggregated_stats[350][:10]


#%%
# Compute random cosine similarities of X columns

n_tries = 10000
cos_sim_random_pairs = t.zeros(n_tries)
for i in trange(n_tries):
    idxs = np.random.choice(X.shape[0], 2, replace=False)
    cos_sim_random_pairs[i] = t.cosine_similarity(X[idxs[0]], X[idxs[1]], dim=-1)
    cos_sim_random_pairs[i] = t.clip(cos_sim_random_pairs[i], -1, 1)
    cos_sim_random_pairs[i] = 1 - t.acos(cos_sim_random_pairs[i]) / np.pi
    # cos_sim_random_pairs[i] = C[idxs[0], idxs[1]]



#%%
# Mean cosine similarity within clusters

# Get the distances
n_clusters = 750
distances = t.tensor(intra_cluster_distances[n_clusters])
dist_weights = t.ones_like(distances) / distances.shape[0]
random_weights = t.ones_like(cos_sim_random_pairs) / n_tries

# Create the figure and subplts
fig, axs = plt.subplots(1, 1, figsize=(8, 6))

# Plot the histogram of mean pairwise distances within clusters
xmax = 1
nbins = 20
axs.hist(distances, bins=nbins, range=(0, xmax), weights=dist_weights, alpha=0.7, label=f'Pairs within clusters ({len(aggregated_stats[n_clusters])} / {n_clusters} clusters\nwith >1 samples considered)')
axs.set_title(f'Angular Distance within clusters vs random datapoints\n (n_clusters={n_clusters})')

# Plot the histogram of pairwise cosine similarities for random pairs
axs.hist(cos_sim_random_pairs, bins=nbins, weights=random_weights, range=(0, xmax), alpha=0.5, label=f'Random pairs ({n_tries} samples)')

axs.set_xlabel('Pairwise angular distance')
axs.set_ylabel('Relative frequency')
axs.legend(loc="upper left")

# Adjust the spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()

#%%
# Mean Distances within clusters over Number of Clusters

# Get the data
mean_distances = [t.tensor(agg)[:, 0].mean() for n_clusters, agg in aggregated_stats.items()]
num_clusters = [t.tensor(agg).shape[0] for n_clusters, agg in aggregated_stats.items()]

# Create the figure and subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot the mean intra-cluster distances
axs[0].scatter(ccfg.cluster_counts, mean_distances)
axs[0].set_xlabel('Total Number of Clusters')
axs[0].set_ylabel('Mean Distances within clusters')
axs[0].set_title('Mean Distances within clusters\n mean over all clusters for a fixed number of total clusters')

# Plot the number of clusters with more than one sample
axs[1].scatter(ccfg.cluster_counts, num_clusters)
axs[1].set_xlabel('Total Number of Clusters')
axs[1].set_ylabel('Number of Clusters with >1 Samples')
axs[1].set_title('Number of Clusters with More than One Sample')

# Adjust the spacing between subplots
plt.tight_layout()

# Show the figure
plt.show()

# %%
# Pairwise clusters vs load similarity matrix test
# # Find pairwise distances in cluster
# n_clusters = 350
# cluster_idx = 0
# cluster_mask = t.tensor(activation_clusters[str(n_clusters)]) == cluster_idx
# cluster_indices = t.where(cluster_mask)[0]
# C_cluster = C[cluster_indices][:, cluster_indices]
# cluster_upper_triangle = select_upper_triangle(C_cluster)
# # test wheter the upper triangle is the same as the pairwise distances
# t.allclose(cluster_upper_triangle, find_pairwise_distances(X[cluster_indices], angular=True))
# %%