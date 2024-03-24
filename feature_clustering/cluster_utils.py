#%%
import numpy as np
import torch as t
import einops
from dataclasses import dataclass
import sys
sys.path.append('../')

@dataclass
class ClusterConfig():
    def __init__(self, 
                 model_name, 
                 loss_threshold, 
                 n_samples,
                 n_ctx, 
                 submodules_generic,
                 dict_size,
                 results_dir=None,
                 cluster_counts=None,
                 n_submodules=None,
                 random_seed=42,
                 dict_id=10
                 ):
        self.model_name = model_name
        self.loss_threshold = loss_threshold
        self.n_samples = n_samples # Not achieving the exact number of samples if n_samples is not divisible by batch_size
        self.n_ctx = n_ctx
        self.submodules_generic = submodules_generic
        self.dict_size = dict_size
        self.dict_id = dict_id
        self.results_dir = results_dir
        self.cluster_counts = cluster_counts
        self.n_submodules = n_submodules
        self.random_seed = random_seed

def row_filter(X, row_idxs):
    # Find the positions in indices[1] that are in pos_idxs
    if not X.is_sparse:
        return X[row_idxs, :]
    row_idxs = t.tensor(row_idxs)
    selected_positions = t.isin(X._indices()[0], row_idxs)

    # Filter the indices and values
    filtered_indices = X._indices()[:, selected_positions]
    filtered_values = X._values()[selected_positions]
    _, inverse_indices = t.unique(filtered_indices[0], sorted=True, return_inverse=True)
    filtered_indices[0] = inverse_indices

    # Create the new sparse tensor
    new_size = t.tensor(X.size())
    new_size[0] = len(row_idxs)
    new_size = t.Size(new_size)
    new_tensor = t.sparse_coo_tensor(filtered_indices, filtered_values, new_size)
    return new_tensor

def pos_filter(X, ccfg, pos_idxs):
    # Filter positions
    ## Dense X
    if not X.is_sparse:
        return X[:, pos_idxs, :]

    ## Sparse X
    # Find the positions in indices[1] that are in pos_idxs
    pos_idxs = t.tensor(pos_idxs)
    selected_positions = t.isin(X._indices()[1], pos_idxs)

    # Filter the indices and values
    filtered_indices = X._indices()[:, selected_positions]
    filtered_values = X._values()[selected_positions]
    # Since we're selecting columns, we need to adjust the indices to be sequential. E.g. if we have pos_idx = [2, 4, 5] we need to adjust them to [0, 1, 2]
    _, inverse_indices = t.unique(filtered_indices[1], sorted=True, return_inverse=True)
    filtered_indices[1] = inverse_indices
    # Now convert [batch, len(pos_idxs), dmodel] to [batch, len(pos_idxs) * dmodel]
    indices_2d = t.zeros((2, filtered_indices.size(1)), dtype=t.long)
    indices_2d[0] = filtered_indices[0]
    indices_2d[1] = filtered_indices[2] + filtered_indices[1] * ccfg.dictionary_size * ccfg.n_submodules

    # Create the new sparse tensor
    new_size = t.Size([X.size(0), len(pos_idxs) * ccfg.dictionary_size * ccfg.n_submodules])
    new_tensor = t.sparse_coo_tensor(indices_2d, filtered_values, new_size)
    return new_tensor

def get_pos_aggregation_description(pos_aggregation):
    if type(pos_aggregation) == int:
        aggregation_description = f"final-{pos_aggregation}-pos"
    elif pos_aggregation == "sum":
        aggregation_description = "sum-over-pos"
    else:
        raise ValueError(f"Invalid pos_aggregation: {pos_aggregation}")
    return aggregation_description

def pattern_matrix_pos_aggregated(X, ccfg, pos_aggregation):
    if type(pos_aggregation) == int:
        pos_idxs = ccfg.n_ctx - t.arange(1, pos_aggregation+1) # ccfg.n_ctx-1 for the final token, or a list of positions to cluster [0, 1, 2, ...
        X_pos_aggregated = pos_filter(X, ccfg, pos_idxs)
        print(f"Final positions considered: {pos_idxs}")
    elif pos_aggregation == "sum":
        X_pos_aggregated = X.sum(dim=1) # sum over positions
    else:
        raise ValueError(f"Invalid pos_aggregation: {pos_aggregation}")
    return X_pos_aggregated, get_pos_aggregation_description(pos_aggregation)

def loss_idx_to_dataset_idx(idx, starting_indexes):
    """given an idx in range(0, 10658635), return
    a document index in range(0, 600000) and pred-in-document
    index in range(0, 1023). Note token-in-document idx is
    exactly pred-in-document + 1"""
    document_index = np.searchsorted(starting_indexes, idx, side="right") - 1
    final_token_in_context_index = idx - starting_indexes[document_index]
    return int(document_index), int(final_token_in_context_index)

def get_tokenized_context_y(ccfg, dataset, doc_idx, final_token_in_context_index):
    """The length of idxs determines the batch size. Given idx in range(0, 10658635)
    and predicted token within sample, in range(1, 1024)."""
    doc_idx = int(doc_idx)
    final_token_in_context_index = int(final_token_in_context_index)
    y_idx = final_token_in_context_index + 1
    context_start_idx = y_idx - ccfg.n_ctx
    context = dataset[doc_idx]['input_ids'][0][context_start_idx:y_idx]
    y = dataset[doc_idx]['input_ids'][0][y_idx]
    return context, y, doc_idx

def get_string_context_y(tokenizer, ccfg, dataset, doc_idx, final_token_in_context_index):
    context, y, doc_idx = get_tokenized_context_y(ccfg, dataset, doc_idx, final_token_in_context_index)
    concatenated_context = "".join(tokenizer.decode(context))
    return concatenated_context, tokenizer.decode(y), doc_idx

#%% 
# Test pos_filter_sparse
# import einops

# batch = 2
# pos = 3
# dmodel = 4
# ccfg = ClusterConfig(model_name="test", loss_threshold=0.1, num_samples=8192, n_ctx=64, submodules_generic=1, dictionary_size=2, results_dir=None, cluster_counts=None, n_submodules=2, random_seed=42)
# pos_idxs = [1, 2]

# X = t.LongTensor([[[0, 0, 1, 2], [2, 7, 7, 9], [0, 1, 2, 3]], [[0, 0, 1, 2], [2, 7, 7, 9], [0, 1, 2, 3]]]) 
# print(f"X shape: {X.shape}")
# # print(f"X indices: {X}")
# print(X[:, pos_idxs, :])

# Xs = X.to_sparse()
# Xs = pos_filter_sparse(Xs, ccfg, pos_idxs)
# print(f"Xs shape: {Xs.shape}")
# print(f"Xs indices: {Xs._indices()}")
# print(f"Xs: {Xs.to_dense()}")

# Xd = einops.rearrange(X[:, pos_idxs, :], 'b p d -> b (p d)')
# print(f"Xd shape: {Xd.shape}")
# print(f"Xd: {Xd}")

# t.allclose(Xs.to_dense(), Xd)

# #%%
# # Test row_filter_sparse

# rows = [0, 2]
# X = t.LongTensor([[0, 0, 1, 2], [2, 7, 7, 9], [0, 1, 2, 3]])
# print(f"X shape: {X.shape}")

# Xs = X.to_sparse()
# Xs = row_filter_sparse(Xs, rows)
# print(f"Xs shape: {Xs.shape}")
# print(f"Xs indices: {Xs._indices()}")
# print(f"Xs: {Xs.to_dense()}")

# Xd = X[rows, :]
# print(f"Xd shape: {Xd.shape}")
# print(f"Xd: {Xd}")

# t.allclose(Xs.to_dense(), Xd)