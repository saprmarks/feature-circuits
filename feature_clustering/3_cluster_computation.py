import json
import os
import torch as t
import numpy as np
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.cluster import SpectralClustering
from cluster_utils import ClusterConfig, pos_filter_sparse

def compute_similarity_matrix(X1, X2, angular=False):
    X1_norm = F.normalize(X1, p=2, dim=-1)
    X2_norm = F.normalize(X2, p=2, dim=-1)
    C = t.matmul(X1_norm, X2_norm.T) # cos sim
    if angular:
        C = t.clamp(C, -1, 1)
        C = 1 - t.acos(C) / np.pi
    return C

def compute_similarity_matrix_blockwise(X, angular=False, block_length=4, device="cuda:0"):
    X = X.to(device)
    C = t.zeros(X.shape[0], X.shape[0], device=device)
    n_blocks = X.shape[0] // block_length
    block_idxs = [t.arange(i*block_length, min((i+1)*block_length, X.shape[0]), device=device) for i in range(n_blocks)]
    for i in range(n_blocks):
        for j in range(n_blocks):
            row_idxs = block_idxs[i]
            col_idxs = block_idxs[j]
            C_block = compute_similarity_matrix(X[row_idxs], X[col_idxs], angular=angular)
            C[row_idxs.unsqueeze(1), col_idxs] = C_block
            if i != j: # fill the transpose of the block if not on diagonal
                C[col_idxs.unsqueeze(1), row_idxs] = C_block.T
    return C

def run_clustering(X, ccfg):
    # Compute the cosine similarity of the activation patterns (n_pos * n_submodules * n_features) per sample
    X = X.to_dense()
    X = X.reshape(X.shape[0], -1)
    print(f'computing cosine similarity of shape {X.shape}')

    C = compute_similarity_matrix_blockwise(X, block_length=1024, angular=True)
    C = C.cpu().numpy()

    # Run spectral clustering
    results = dict()
    for n_clusters in tqdm(ccfg.cluster_counts):
        clusters_labels = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', n_init=30, random_state=ccfg.random_seed).fit_predict(C)
        results[n_clusters] = clusters_labels.tolist()
    return results


if __name__ == "__main__":

    # Load config and add cluster counts
    results_dir = "/home/can/feature_clustering/clustering_pythia-70m-deduped_tloss0.1_nsamples8192_npos64_filtered-induction_mlp-attn-resid"
    ccfg = ClusterConfig(**json.load(open(os.path.join(results_dir, "config.json"), "r")))

    # Define Experiment variables
    data_source = "activations" # activations or lin_effects
    CLUSTER_COUNTS = [10, 50, 100, 200, 350, 500, 750, 1000]
    device = "cuda:0"
    pos_idxs = [ccfg.n_pos - 1] # ccfg.n_pos-1 for the final token, or a list of positions to cluster [0, 1, 2, ...
    pos_string = "final" if pos_idxs == [ccfg.n_pos - 1] else "_".join([str(pos) for pos in pos_idxs])

    # Add cluster counts to config and save
    ccfg.cluster_counts = CLUSTER_COUNTS
    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(ccfg.__dict__, f)

    # Load and filter data
    X = t.load(os.path.join(results_dir, data_source + ".pt"))
    print(f"Loaded {data_source} of shape {X.shape}")
    X_pos_filtered = pos_filter_sparse(X, ccfg, pos_idxs)
    print(f"Filtered {data_source} of shape {X_pos_filtered.shape}")
    # X_pos_filtered = X_pos_filtered.to(device) # sklearn.SpectralClustering only with numpy on cpu

    # Run clustering
    clusters = run_clustering(X_pos_filtered, ccfg)

    # Save results using json
    with open(os.path.join(results_dir, f"clusters_{data_source}_pos{pos_string}.json"), "w") as f:
        json.dump(clusters, f)
    
# %%
# test for comparing functions compute_similarity_matrix_blockwise and compute_similarity_matrix
# X = t.randn(100, 1024)
# C_blockwise = compute_similarity_matrix_blockwise(X, angular=False, block_length=4, device="cpu")
# C = compute_similarity_matrix(X, X, angular=False)
# assert t.allclose(C_blockwise, C, atol=1e-5)
