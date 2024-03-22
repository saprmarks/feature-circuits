# %%
import json
import os
import numpy as np
import torch as t
from tqdm import tqdm

model_name = "pythia-70m-deduped"
n_feats_per_submod = 512 * 64
n_submod = 6
n_total_feats = n_feats_per_submod * n_submod
random_seed = 42

device = "cuda:0"
num_svd_components = 900 # With 1024 components and percentage 0.9, truncation dim is 805
singular_value_cutoff = 0.95
CLUSTER_COUNTS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 500, 1000]

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

# Load feature activations and gradients on 1k contexts
act_grad_filename = f"act-n-grad-cat_{param_summary}.json"
act_per_context = json.load(open(os.path.join(activations_dir, act_grad_filename), "r"))

y_global_idx = np.array(list(act_per_context.keys()), dtype=int)
num_y = len(act_per_context)


# %%
# Sweep over score metrics and SVD
for score_metric in ["act", "act-grad"]:
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
        # Perform full SVD
        # num_svd_components = min(X.shape) # non-truncated SVD

        U, S, V = t.svd_lowrank(X, q=num_svd_components, niter=3)

        # Save SVD results with torch.save
        svd_filename = f"svd-comp{num_svd_components}_{score_metric}_{param_summary}.pt"
        t.save((U, S, V), os.path.join(svd_dir, svd_filename))

        U, S, V = t.load(os.path.join(svd_dir, svd_filename))
