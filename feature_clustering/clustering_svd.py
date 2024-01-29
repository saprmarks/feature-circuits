# %%
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import torch as t
from tqdm import tqdm
from sklearn.cluster import SpectralClustering

model_name = "pythia-70m-deduped"
n_feats_per_submod = 512 * 64
n_submod = 6
n_total_feats = n_feats_per_submod * n_submod
random_seed = 42

device = "cuda:0"
num_svd_components = 900 # With 1024 components and percentage 0.9, truncation dim is 805
singular_value_cutoff = 0.95
CLUSTER_COUNTS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 500, 1000, 1500]

activations_dir = "/home/can/feature_clustering/activations"
clusters_dir = "/home/can/feature_clustering/clusters"
svd_dir = "/home/can/feature_clustering/svd"
loss_threshold = 0.03
skip = 512
num_tokens = 10000
feature_pattern_reduction_across_positions = "sum" # "sum" or "pos"
n_pos = 10
submod_type_names = "mlp"
param_summary = f"{model_name}_tloss{loss_threshold}_ntok{num_tokens}_skip{skip}_npos{n_pos}_{submod_type_names}"

# %%
# Load feature activations and gradients on 1k contexts
act_grad_filename = f"act-n-grad-99_{param_summary}.json"
act_per_context = json.load(open(os.path.join(activations_dir, act_grad_filename), "r"))

y_global_idx = np.array(list(act_per_context.keys()), dtype=int)
num_y = len(act_per_context)


# %%
# Sweep over score metrics and SVD
for score_metric in ["act", "act-grad"]:
    for do_svd in [True, False]:
        print(f"\nScore metric: {score_metric}, SVD: {do_svd}\n")

        # Load feature activations and gradients on 1k contexts
        if feature_pattern_reduction_across_positions == "sum":
            X = t.zeros((num_y, n_total_feats))
            for row, context in tqdm(enumerate(act_per_context), desc="Loading into matrix, Row", total=num_y):
                for col, act, grad in act_per_context[context][feature_pattern_reduction_across_positions]:
                    col = int(col)
                    if score_metric == "act":
                        X[row, col] = act
                    elif score_metric == "act-grad":
                        X[row, col] = act * grad
                    else:
                        raise ValueError("Unknown score_metric")
            X.to_sparse().to(device)
            print(f'X shape: {X.shape}')
        elif feature_pattern_reduction_across_positions == "pos":
            X = t.zeros((num_y, n_total_feats * n_pos))
            for row, context in tqdm(enumerate(act_per_context), desc="Loading into matrix, Row", total=num_y):
                for pos_idx, col, act, grad in act_per_context[context][feature_pattern_reduction_across_positions]:
                    col = int(col)
                    pos_idx = int(pos_idx)
                    if score_metric == "act":
                        X[row, col + (pos_idx+n_pos) * n_total_feats] = act
                    elif score_metric == "act-grad":
                        X[row, col + (pos_idx+n_pos) * n_total_feats] = act * grad
                    else:
                        raise ValueError("Unknown score_metric")
            X.to_sparse().to(device)
            print(f'X shape: {X.shape}')
        else:
            raise ValueError("Unknown feature_pattern_reduction_across_positions")

        truncation_dim = 'None'
        if do_svd:
            # Perform full SVD
            # num_svd_components = min(X.shape) # non-truncated SVD

            U, S, V = t.svd_lowrank(X, q=num_svd_components, niter=3)

            # Save SVD results with torch.save
            svd_filename = f"svd-comp{num_svd_components}_{score_metric}_{param_summary}.pt"
            t.save((U, S, V), os.path.join(svd_dir, svd_filename))

            # Plot singular values
            # plt.plot(svd.singular_values_)
            # plt.title("Singular values")
            # plt.yscale("log")
            # plt.show()

            # Truncate 
            def find_cutoff_index_numpy(a):
                # Step 1: Normalize the array
                normalized_a = a / np.sum(a)

                # Step 2: Find the cutoff index using np.cumsum
                cumulative_sum = np.cumsum(normalized_a)

                # Find the index where cumulative sum exceeds or equals 90%
                cutoff_index = np.argmax(cumulative_sum >= singular_value_cutoff)

                # Step 3: Return the cutoff index
                return cutoff_index

            truncation_dim = find_cutoff_index_numpy(S.cpu().numpy())
            print(f"Truncation dim: {truncation_dim}")
            X_post = t.matmul(X, V[:, :truncation_dim])
            X_post.to_sparse().to(device)
        else:
            X_post = X

        # Using scipy sparse matrix

        # Norm with pt
        X_norm = t.norm(X_post, dim=1, keepdim=True)
        X_norm_abs = X_norm.abs()

        # Dot product
        C = t.matmul(X_norm, X_norm.T)
        C_abs = t.matmul(X_norm_abs, X_norm_abs.T)

        # Clip
        C = t.clamp(C, -1, 1)
        C_abs = t.clamp(C_abs, -1, 1)

        # Convert to radians=cosine distance
        C = 1 - t.acos(C) / np.pi
        C_abs = 1 - t.acos(C_abs) / np.pi


        # Run spectral clustering
        # Maybe have to convert to numpy?

        results = dict()

        for n_clusters in tqdm(CLUSTER_COUNTS):
            clusters_labels = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', n_init=30, random_state=random_seed).fit_predict(C)
            clusters_labels_abs = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', n_init=30, random_state=random_seed).fit_predict(C_abs)
            results[n_clusters] = (clusters_labels.tolist(), clusters_labels_abs.tolist())


        # Save results using json
        results_filename = f"clusters_{score_metric}_svd-comp{truncation_dim}-cut{singular_value_cutoff}_{param_summary}.json"
        with open(os.path.join(clusters_dir, results_filename), "w") as f:
            json.dump(results, f)
# %%