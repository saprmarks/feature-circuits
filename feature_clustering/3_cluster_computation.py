import json
import os
import torch as t
import numpy as np
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.cluster import SpectralClustering
from cluster_utils import ClusterConfig, pos_filter_sparse

def run_clustering(X, ccfg):
    # Compute the cosine similarity of the activation patterns (n_pos * n_submodules * n_features) per sample
    X = X.to_dense()
    X = X.reshape(X.shape[0], -1)
    print(f'computing cosine similarity of shape {X.shape}')
    X_norm = F.normalize(X, p=2, dim=-1) # [n_samples, n_features]
    C = t.matmul(X_norm, X_norm.T) # [n_samples, n_samples]
    C = t.clamp(C, -1, 1)
    C = 1 - t.acos(C) / np.pi
    C = C.cpu().numpy()

    # Run spectral clustering
    results = dict()
    for n_clusters in tqdm(ccfg.cluster_counts):
        clusters_labels = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', n_init=30, random_state=ccfg.random_seed).fit_predict(C)
        results[n_clusters] = clusters_labels.tolist()
    return results


if __name__ == "__main__":

    results_dir = "/home/can/feature_clustering/clustering_pythia-70m-deduped_tloss0.1_nsamples1024_npos32_filtered-induction_mlp-attn-resid"
    data_source = "lin_effects" # activations or lin_effects
    device = "cuda:0"
    CLUSTER_COUNTS = [10, 50, 75, 100, 200, 350, 500, 750, 1000]

    # Load config
    ccfg = ClusterConfig(**json.load(open(os.path.join(results_dir, "config.json"), "r")))
    ccfg.cluster_counts = CLUSTER_COUNTS
    # save the config
    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(ccfg.__dict__, f)

    pos_idxs = [ccfg.n_pos - 1] # ccfg.n_pos-1 for the final token, or a list of positions to cluster [0, 1, 2, ...
    pos_string = "final" if pos_idxs == [ccfg.n_pos - 1] else "_".join([str(pos) for pos in pos_idxs])

    # Load data
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
