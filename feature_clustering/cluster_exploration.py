#%%
# Imports

import json
import os
import numpy as np
import torch as t
import torch.nn.functional as F
from collections import defaultdict
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from cluster_utils import ClusterConfig, pos_filter_sparse;

#%%
# Config

results_dir = "/home/can/feature_clustering/clustering_pythia-70m-deduped_tloss0.1_nsamples1024_npos32_filtered-induction_mlp-attn-resid"
device = "cuda:0"
ccfg = ClusterConfig(**json.load(open(os.path.join(results_dir, "config.json"), "r")))

#%%
# Helper functions

def load_pattern_matrix_final_pos(results_dir, data_source, ccfg):
    pos_idxs = [ccfg.n_pos - 1]
    X = t.load(os.path.join(results_dir, data_source + ".pt"))
    X = pos_filter_sparse(X, ccfg, pos_idxs)
    X = X.to_dense().reshape(X.shape[0], -1)
    return X

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
########## Activations

data_source = "activations" # activations or lin_effects
cluster_filename = f"clusters_activations_posfinal.json"
activation_clusters = json.load(open(os.path.join(results_dir, cluster_filename), "r"))
X_activations = load_pattern_matrix_final_pos(results_dir, data_source, ccfg)

#%%
# Inspect single cluster

n_clusters = 350
cluster_idx = 6

act_clusters = t.tensor(activation_clusters[str(n_clusters)])
cluster_indices = t.where(act_clusters == cluster_idx)[0]
pairwise_distances = find_pairwise_distances(X_activations[cluster_indices], angular=False)

print(f'cluster {n_clusters}/{cluster_idx} has {cluster_indices.shape[0]} samples')
print(f'unique distances {pairwise_distances.unique()}\n\n\n')


# Print contexts in clusters
samples = json.load(open(os.path.join(results_dir, "samples.json"), "r"))
for idx in cluster_indices[:10]:
    idx = str(idx.item())
    print(f'index: {idx}')
    print(f'context: {samples[idx]["context"]}')
    print(f'answer: {samples[idx]["answer"]}')
    print('---\n\n')


#%%
# Compute intra-cluster distances for all n_clusters

aggregated_intra_cluster_distances = defaultdict(list)
for i, n_clusters in enumerate(ccfg.cluster_counts):
    for cluster_idx in range(n_clusters):
        cluster_mask = t.tensor(activation_clusters[str(n_clusters)]) == cluster_idx
        if cluster_mask.sum() > 1:
            pairwise_distances = find_pairwise_distances(X_activations[cluster_mask], angular=False)
            aggregated_intra_cluster_distances[n_clusters].append((pairwise_distances.mean(), cluster_idx, cluster_mask.sum()))

print(f'columns: mean distance, cluster_idx, n_samples in cluster')
aggregated_intra_cluster_distances[350][:10]

#%%
# Compute random cosine similarities of X columns

n_tries = 10000
cos_sim_random_pairs = t.zeros(n_tries)
for i in range(n_tries):
    idxs = np.random.choice(X_activations.shape[0], 2, replace=False)
    cos_sim_random_pairs[i] = t.cosine_similarity(X_activations[idxs[0]], X_activations[idxs[1]], dim=-1)



#%%
# Mean cosine similarity within clusters for n_clusters=100

# Get the distances
n_clusters = 500
distances = t.tensor(aggregated_intra_cluster_distances[n_clusters])[:, 0]

# Create the figure and subplts
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot the histogram of mean pairwise distances within clusters
xmax = 1.05
nbins = 20
axs[0].hist(distances, bins=nbins, range=(0, xmax))
axs[0].set_xlabel('Mean Pairwise Cosine Similarity within Cluster')
axs[0].set_ylabel('Frequency')
axs[0].set_title(f'Mean Cosine Similarities within clusters\n (n_clusters={n_clusters}, only {distances.shape[0]} clusters with >1 samples considered)')



# Plot the histogram of pairwise cosine similarities for random pairs
axs[1].hist(cos_sim_random_pairs, bins=nbins , range=(0, xmax))
axs[1].set_xlabel('Pairwise Cosine similarity')
axs[1].set_ylabel('Frequency')
axs[1].set_title(f'Cosine Similarity of {n_tries} random pairs from dataset')

# Adjust the spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()

#%%
# Mean Distances within clusters over Number of Clusters

# Get the data
mean_distances = [t.tensor(agg)[:, 0].mean() for n_clusters, agg in aggregated_intra_cluster_distances.items()]
num_clusters = [t.tensor(agg).shape[0] for n_clusters, agg in aggregated_intra_cluster_distances.items()]

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
