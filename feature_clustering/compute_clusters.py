import json
import os
import torch as t
import numpy as np
from torch.nn import functional as F
from tqdm import tqdm, trange
from sklearn.cluster import SpectralClustering
from cluster_utils import ClusterConfig, row_filter, pattern_matrix_pos_aggregated
import argparse
from collections import defaultdict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute similarity matrices of feature activations and linear effects")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for data loader")
    parser.add_argument("--results_dir", type=str, default="/home/can/feature_clustering/clustering_pythia-70m-deduped_tloss0.1_nsamples32768_npos16_filtered-induction_attn-mlp-resid", help="Directory to load data and save results")
    parser.add_argument("--similarity_matrix_dir", type=str, default=None, help="Directory to load similarity matrix")
    parser.add_argument("--aggregation", type=str, default=None, help="Method for aggregating positional information ('sum' or int x for final x positions)")
    parser.add_argument("--component_type", choices=["sparse_features", "dense_neurons"], help="Type of components to cluster (sparse features or dense neurons)")
    parser.add_argument("--cluster_counts", type=int, nargs="+", default=[2000, 4000, 8000], help="Number of clusters to use for spectral clustering")
    args = parser.parse_args()

    # Add cluster counts to config and save
    ccfg = ClusterConfig(**json.load(open(os.path.join(args.results_dir, "config.json"), "r")))
    ccfg.cluster_counts = args.cluster_counts
    with open(os.path.join(args.results_dir, "config.json"), "w") as f:
        json.dump(ccfg.__dict__, f)

    # Load similarity matrix
    similarity_matrix = t.load(args.similarity_matrix_dir)

    # Run spectral clustering
    results = dict()
    for n_clusters in tqdm(ccfg.cluster_counts, desc="Clustering"):
        clusters_labels = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', n_init=30, random_state=ccfg.random_seed).fit_predict(similarity_matrix)

        result_dict = defaultdict(list)
        for i, cluster_label in enumerate(clusters_labels):
            result_dict[cluster_label].append(i)
        result_dict = dict(sorted(result_dict.items(), key=lambda x: len(x[1]), reverse=True))

        # Save results using json
        cluster_filename = f"cluster_{args.component_type}_nclusters{n_clusters}_nsamples{ccfg.n_samples}_nctx{ccfg.n_ctx}_agg{args.aggregation}.json"
        with open(os.path.join(args.results_dir, cluster_filename), "w") as f:
            json.dump(result_dict, f)