# %%
import json
import os
import numpy as np
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import SpectralClustering
import pickle
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

n_feats_per_submod = 512 * 64
n_submod = 6
n_total_feats = n_feats_per_submod * n_submod
random_seed = 42

results_dir = "/home/can/feature_clustering/results"
model_name = "pythia-70m-deduped"
loss_threshold = 0.005
skip = 50
num_tokens = 10000
k = "nonzero"
submod_type_names = "mlp"
param_string = f"{model_name}_loss-thresh{loss_threshold}_skip{skip}_ntok{num_tokens}_{k}_{submod_type_names}"

# %%
# Load feature activations and gradients on 1k contexts
act_grad_filename = f"act-n-grad_{param_string}.json"
act_per_context = json.load(open(os.path.join(results_dir, act_grad_filename), "r"))

y_global_idx = np.array(list(act_per_context.keys()), dtype=int)
num_y = len(act_per_context)


# %%
# Sweep over score metrics and SVD
for score_metric in ["act", "act-grad"]:
    for do_svd in [True, False]:
        print(f"###### Score metric: {score_metric}, SVD: {do_svd}")

        # %%
        # Load into matrix
        X = np.zeros((len(act_per_context), n_total_feats))
        for row, context in tqdm(enumerate(act_per_context), desc="Loading into matrix, Row", total=num_y):
            for col, act, grad in act_per_context[context]:
                col = int(col)
                if score_metric == "act":
                    X[row, col] = act
                elif score_metric == "act-grad":
                    X[row, col] = act * grad
                else:
                    raise ValueError("Unknown score_metric")

        print(f'X shape: {X.shape}')

        truncation_dim = 'None'
        C, C_abs = None, None
        if do_svd:
            # %%
            # Perform full SVD
            # num_svd_components = min(X.shape) # non-truncated SVD
            num_svd_components = 1024 # With 1024 components and percentage 0.9, truncation dim is 805

            svd = TruncatedSVD(n_components=num_svd_components, random_state=random_seed)
            svd.fit(csr_matrix(X)) # SVD speedup with sparse matrix representation, tested: eigenvalues are eqivalent with dense matrix
            svd.components_.shape

            # %%
            # Save SVD results
            svd_dir = "/home/can/feature_clustering/cache"
            svd_filename = f"svd-comp{num_svd_components}_{score_metric}_{param_string}.pkl"
            with open(os.path.join(svd_dir, svd_filename), "wb") as f:
                pickle.dump(svd, f)

            # %%
            # Plot singular values
            # plt.plot(svd.singular_values_)
            # plt.title("Singular values")
            # plt.yscale("log")
            # plt.show()

            # %%
            # Truncate 
            def find_cutoff_index_numpy(a):
                # Step 1: Normalize the array
                normalized_a = a / np.sum(a)

                # Step 2: Find the cutoff index using np.cumsum
                cumulative_sum = np.cumsum(normalized_a)
                cutoff_percentage = 0.9

                # Find the index where cumulative sum exceeds or equals 90%
                cutoff_index = np.argmax(cumulative_sum >= cutoff_percentage)

                # Step 3: Return the cutoff index
                return cutoff_index

            truncation_dim = find_cutoff_index_numpy(svd.singular_values_)
            print(f"Truncation dim: {truncation_dim}")
            V = svd.components_.T[:, :truncation_dim]
            X_trunc = np.dot(X, V)

            # Norm
            X_trunc_norm = normalize(X, norm='l2', axis=1)
            X_trunc_norm_abs = np.abs(X_trunc_norm)

            # Dot product
            C = X_trunc_norm.dot(X_trunc_norm.T)
            C_abs = X_trunc_norm_abs.dot(X_trunc_norm_abs.T)

        else:
            # %%
            # Using scipy sparse matrix
            # Does using csr give speedup when using svd?   
            X_csr = csr_matrix(X)

            # Norm
            X_csr_norm = normalize(X_csr, norm='l2', axis=1)
            X_csr_norm_abs = np.abs(X_csr_norm)

            # Dot product
            C = X_csr_norm.dot(X_csr_norm.T).toarray()
            C_abs = X_csr_norm_abs.dot(X_csr_norm_abs.T).toarray()

        # Clip
        C = np.clip(C, -1, 1)
        C_abs = np.clip(C_abs, -1, 1)

        # Convert to radians=cosine distance
        C = 1 - np.arccos(C) / np.pi
        C_abs = 1 - np.arccos(C_abs) / np.pi


        # %%
        # Run spectral clustering
        CLUSTER_COUNTS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 500, 1000, 1500]

        results = dict()

        for n_clusters in tqdm(CLUSTER_COUNTS):
            clusters_labels = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', n_init=30, random_state=random_seed).fit_predict(C)
            clusters_labels_abs = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', n_init=30, random_state=random_seed).fit_predict(C_abs)
            results[n_clusters] = (clusters_labels.tolist(), clusters_labels_abs.tolist())


        # %%
        # Save results using json
        results_dir = "/home/can/feature_clustering/clusters"
        results_filename = f"clusters_{score_metric}_svd-comp{truncation_dim}_{param_string}.json"
        with open(os.path.join(results_dir, results_filename), "w") as f:
            json.dump(results, f)