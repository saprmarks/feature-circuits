import json
import os
import torch as t
import numpy as np
from torch.nn import functional as F
from tqdm import tqdm, trange
from sklearn.cluster import SpectralClustering
from cluster_utils import ClusterConfig, row_filter, pattern_matrix_pos_aggregated

def compute_similarity_matrix(X1, X2, angular=False):
    X1, X2 = X1.to_dense(), X2.to_dense() # norm_sparse currently only supports full reductions, so 'dim' must either be empty or contain all dimensions of the input
    X1_norm = F.normalize(X1, p=2, dim=-1)
    X2_norm = F.normalize(X2, p=2, dim=-1)
    C = t.matmul(X1_norm, X2_norm.T) # cos sim
    if angular:
        C = t.clamp(C, -1, 1)
        C = 1 - t.acos(C) / np.pi
    return C

def compute_similarity_matrix_blockwise(X, angular=False, absolute=False, block_length=4, device="cuda:0"):
    X = X.to(device)
    C = t.zeros(X.shape[0], X.shape[0], device=device, dtype=t.float16)
    n_blocks = X.shape[0] // block_length
    block_idxs = [t.arange(i*block_length, min((i+1)*block_length, X.shape[0]), device=device) for i in range(n_blocks)]
    for i in trange(n_blocks):
        for j in range(n_blocks):
            row_idxs = block_idxs[i]
            col_idxs = block_idxs[j]
            X_row = row_filter(X, row_idxs)
            X_col = row_filter(X, col_idxs)
            C_block = compute_similarity_matrix(X_row, X_col, angular=angular)
            if absolute:
                C_block = t.abs(C_block)
            C[row_idxs.unsqueeze(1), col_idxs] = C_block
            if i != j: # fill the transpose of the block if not on diagonal
                C[col_idxs.unsqueeze(1), row_idxs] = C_block.T
    return C

def run_clustering(X, ccfg, block_length=1024, device="cuda:0", aggregation_description="", data_source=""):
    # Compute the cosine similarity of the activation patterns (n_pos * n_submodules * n_features) per sample
    print(f'computing cosine similarity of shape {X.shape}')
    C = compute_similarity_matrix_blockwise(X, block_length=block_length, angular=True, absolute=False, device=device)
    C = C.cpu().numpy()

    # Run spectral clustering
    print(f"starting clustering with {ccfg.cluster_counts} clusters")
    results = dict()
    for n_clusters in tqdm(ccfg.cluster_counts):
        clusters_labels = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', n_init=30, random_state=ccfg.random_seed).fit_predict(C)
        results[n_clusters] = clusters_labels.tolist()
    
    C = t.tensor(C) # For saving
    return results, C


if __name__ == "__main__":

    # Define Experiment variables
    results_dir = "/home/can/feature_clustering/clustering_pythia-70m-deduped_tloss0.1_nsamples16384_npos128_filtered-induction_attn-mlp-resid"
    CLUSTER_COUNTS = [100, 500, 1000, 2500, 5000, 7500, 10000]#[10, 50, 100, 200, 350, 500, 750, 1000, 1250]
    is_dense = False
    data_source = "activations" # activations or lin_effects
    pos_aggregation = "1" # "sum" or int x for final x positions
    block_length = 512
    device = "cuda:0"

    # Add cluster counts to config and save
    ccfg = ClusterConfig(**json.load(open(os.path.join(results_dir, "config.json"), "r")))
    ccfg.cluster_counts = CLUSTER_COUNTS
    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(ccfg.__dict__, f)
    if is_dense:
        dense_name = "dense_"
    else:
        dense_name = ""

    # Load data and perform aggregation over positions
    X_pos_aggregated = t.load(os.path.join(results_dir, dense_name + data_source + ".pt"))
    aggregation_description = "finalpos"
    # X_pos_aggregated, aggregation_description = pattern_matrix_pos_aggregated(X, ccfg, pos_aggregation)
    print(f"Loaded {data_source} of shape {X_pos_aggregated.shape}")

    # Run clustering
    clusters, similarity_matrix= run_clustering(X_pos_aggregated, ccfg, block_length=block_length, aggregation_description=aggregation_description, data_source=data_source)

    # Save results using json
    cluster_filename = dense_name + f"clusters_{data_source}_nsamples{ccfg.num_samples}_nctx{ccfg.n_pos}_{aggregation_description}.json"
    with open(os.path.join(results_dir, cluster_filename), "w") as f:
        json.dump(clusters, f)
    similarity_matrix_filename = dense_name + f"similarity_matrix_{data_source}_nsamples{ccfg.num_samples}_nctx{ccfg.n_pos}_{aggregation_description}.pt"
    t.save(similarity_matrix, os.path.join(results_dir, similarity_matrix_filename))
    
# %%
# test for comparing functions compute_similarity_matrix_blockwise and compute_similarity_matrix
# X = t.randn(100, 1024)
# C_blockwise = compute_similarity_matrix_blockwise(X, angular=False, block_length=4, device="cpu")
# C = compute_similarity_matrix(X, X, angular=False)
# assert t.allclose(C_blockwise, C, atol=1e-5)
