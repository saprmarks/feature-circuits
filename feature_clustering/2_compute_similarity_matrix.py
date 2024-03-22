"""
This script computes the similarity matrices of
1) Feature activations
2) Linear effects of feature on the cross entropy for the next token prediction
across contexts in a dataset.
"""


# Imports and constants
import os
import torch
import torch as t
from torch.nn import functional as F
from nnsight import LanguageModel
from tqdm import trange
import json
import datasets
import gc
import einops
import numpy as np

import sys
sys.path.append('/home/can/dictionary-circuits/')
from loading_utils import load_submodules_and_dictionaries_from_generic, DictionaryCfg
from cluster_utils import ClusterConfig, get_tokenized_context_y, row_filter
from dictionary_learning import AutoEncoder
import argparse









n_batches = ccfg.num_samples // batch_size # Not achieving the exact number of samples if num_samples is not divisible by batch_size
ccfg.n_submodules = len(submodule_names[0]) * model.config.num_hidden_layers
# save n_submodules to the config
with open(os.path.join(results_dir, "config.json"), "w") as f:
    json.dump(ccfg.__dict__, f)

# Data loader
def data_loader(final_token_idxs, batch_size):
    contexts, ys = t.zeros((n_batches, batch_size, ccfg.n_pos)), t.zeros((n_batches, batch_size, 1), dtype=t.int, device=device)
    for i in range(n_batches):
        for j in range(batch_size):
            sample_idx = i * batch_size + j
            context, y, _ = get_tokenized_context_y(
                ccfg, 
                dataset, 
                doc_idx=int(final_token_idxs[sample_idx, 0]), 
                final_token_in_context_index=int(final_token_idxs[sample_idx, 1])
                )
            contexts[i,j] = t.tensor(context).to(device)
            ys[i,j] = t.tensor(y).to(device)
    return contexts.int(), ys.int()
contexts, ys = data_loader(final_token_idxs, batch_size)

# Metric
def metric_fn(logits, target_token_id): # logits shape: (batch_size, seq_len, vocab_size)
    m = torch.log_softmax(logits[:, -1, :], dim=-1) # batch_size, vocab_size
    m = m[t.arange(m.shape[0]), target_token_id] # batch_size
    return m.sum()

# Cache feature activations and gradients and compute similarity blockwise
def cache_activations_and_effects(contexts, ys):

    activations = t.zeros((ccfg.n_submodules, batch_size, ccfg.n_pos, ccfg.dictionary_size), dtype=t.float32, device=device)
    gradients = t.zeros((ccfg.n_submodules, batch_size, ccfg.n_pos, ccfg.dictionary_size), dtype=t.float32, device=device)

    with model.invoke(contexts, fwd_args={'inference': False}) as invoker:
        for layer in range(model.config.num_hidden_layers):
            for i, (sm, ae) in enumerate(zip(submodules[layer], dictionaries[layer])):
                x = sm.output
                is_resid = (type(x.shape) == tuple)
                if is_resid:
                    x = x[0]
                f = ae.encode(x)
                activations[i] = f.detach().save()
                gradients[i] = f.grad.detach().save() # [batch_size, seq_len, vocab_size]

                x_hat = ae.decode(f)
                residual = (x - x_hat).detach()
                if is_resid:
                    sm.output[0][:] = x_hat + residual
                else:
                    sm.output = x_hat + residual
        logits = model.embed_out.output # [batch_size, seq_len, vocab_size]
        metric_fn(logits=logits, target_token_id=ys).backward()
        
    activations = einops.rearrange(activations, 'n_submodules batch_size n_pos dictionary_size -> batch_size n_pos (n_submodules dictionary_size)')
    gradients = einops.rearrange(gradients, 'n_submodules batch_size n_pos dictionary_size -> batch_size n_pos (n_submodules dictionary_size)')
    print(f"nonzero activations: {t.nonzero(activations).shape[0]}")

    # Aggregation of positional information
    if aggregation_description == "sum":
        activations = activations.sum(dim=1)
        gradients = gradients.sum(dim=1)
    elif isinstance(aggregation_description, int):
        if aggregation_description < 0 or aggregation_description > ccfg.n_pos:
            raise ValueError("Invalid aggregation description.")
        activations = activations[:, -aggregation_description:]
        gradients = gradients[:, -aggregation_description:]
    else:
        raise ValueError("Invalid aggregation description.")
    
    # Calculate linear effects
    lin_effects = (activations * gradients) # This elementwise mult would not be faster when setting activations to sparse, as gradients are dense.

    return activations, lin_effects

def compute_similarity_matrix(X1, X2, angular=False):
    X1_norm = F.normalize(X1, p=2, dim=-1)
    X2_norm = F.normalize(X2, p=2, dim=-1)
    C = t.matmul(X1_norm, X2_norm.T) # cos sim
    if angular:
        C = t.clamp(C, -1, 1)
        C = 1 - t.acos(C) / np.pi
    return C

def compute_similarity_matrix_blockwise(angular=False, absolute=False, block_length=4, device="cuda:0"):
    torch.cuda.empty_cache()
    gc.collect()

    C_act = t.zeros(ccfg.num_samples, ccfg.num_samples, device=device, dtype=t.float32)
    C_lin = t.zeros_like(C_act)
    n_blocks = ccfg.num_samples // block_length
    for i in trange(n_blocks, desc="Computing similarity blockwise, ROWS"):
        for j in trange(n_blocks, desc="COLS"):
            
            print(f"GPU Allocated memory: {torch.cuda.memory_allocated(device)/1024**2 :.2f} MB")
            print(f"GPU Cached memory: {torch.cuda.memory_reserved(device)/1024**2 :.2f} MB")
            X_row_act, X_row_lin = cache_activations_and_effects(contexts[i], ys[i])
            X_col_act, X_col_lin = cache_activations_and_effects(contexts[j], ys[j])
            C_block_act = compute_similarity_matrix(X_row_act, X_col_act, angular=angular)
            C_block_lin = compute_similarity_matrix(X_row_lin, X_col_lin, angular=angular)
            if absolute:
                C_block_act = t.abs(C_block_act)
                C_block_lin = t.abs(C_block_lin)
            C_act[i*block_length:(i+1)*block_length, j*block_length:(j+1)*block_length] = C_block_act
            C_lin[i*block_length:(i+1)*block_length, j*block_length:(j+1)*block_length] = C_block_lin
            if i != j: # fill the transpose of the block if not on diagonal
                C_act[j*block_length:(j+1)*block_length, i*block_length:(i+1)*block_length] = C_block_act.T
                C_lin[j*block_length:(j+1)*block_length, i*block_length:(i+1)*block_length] = C_block_lin.T
    return C_act, C_lin

act_similarity_matrix, lin_similarity_matrix = compute_similarity_matrix_blockwise(angular=True, absolute=False, block_length=batch_size, device=device)
act_similarity_matrix_filename = f"similarity_matrix_activations_nsamples{ccfg.num_samples}_nctx{ccfg.n_pos}_{aggregation_description}.pt"
lin_similarity_matrix_filename = f"similarity_matrix_lin_effects_nsamples{ccfg.num_samples}_nctx{ccfg.n_pos}_{aggregation_description}.pt"
t.save(act_similarity_matrix, os.path.join(results_dir, act_similarity_matrix_filename))
t.save(act_similarity_matrix, os.path.join(results_dir, act_similarity_matrix_filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute similarity matrices of feature activations and linear effects")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for data loader")
    parser.add_argument("--results_dir", type=str, default="/home/can/feature_clustering/clustering_pythia-70m-deduped_tloss0.1_nsamples32768_npos16_filtered-induction_attn-mlp-resid", help="Directory to load data and save results")
    parser.add_argument("--tokenized_dataset_dir", type=str, default="/home/can/data/pile_test_tokenized_600k/", help="Directory of tokenized dataset")
    parser.add_argument("--aggregation", type=str, default="sum", help="Method for aggregating positional information ('sum' or int x for final x positions)")
    parser.add_argument("--dict_path", type=str, default="/share/projects/dictionary_circuits/autoencoders/pythia-70m-deduped", help="Directory of dictionaries")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for computation")
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--dict_id", type=int, default=10, help="Dictionary id")
    args = parser.parse_args()


    # Load config, data, model
    ccfg = ClusterConfig(**json.load(open(os.path.join(args.results_dir, "config.json"), "r")))
    final_token_idxs = torch.load(os.path.join(args.results_dir, f"final_token_idxs-tloss{ccfg.loss_threshold}-nsamples{ccfg.n_samples}-nctx{ccfg.n_pos}.pt"))
    dataset = datasets.load_from_disk(args.tokenized_dataset_dir)
    model = LanguageModel("EleutherAI/"+ccfg.model_name, device_map=args.device)

    attns = [layer.attention for layer in model.gpt_neox.layers]
    mlps = [layer.mlp for layer in model.gpt_neox.layers]
    resids = [layer for layer in model.gpt_neox.layers]

    dictionaries = {}
    for i in range(len(model.gpt_neox.layers)):
        ae = AutoEncoder(args.d_model, ccfg.dict_size).to(args.device)
        ae.load_state_dict(t.load(os.path.join(args.dict_path, f'embed/{dict_id}_{ccfg.dict_size}/ae.pt')))
        dictionaries[embed] = ae

        ae = AutoEncoder(args.d_model, ccfg.dict_size).to(args.device)
        ae.load_state_dict(t.load(os.path.join(args.dict_path, f'attn_out_layer{i}/{dict_id}_{ccfg.dict_size}/ae.pt')))
        dictionaries[attns[i]] = ae

        ae = AutoEncoder(args.d_model, ccfg.dict_size).to(args.device)
        ae.load_state_dict(t.load(os.path.join(args.dict_path, f'mlp_out_layer{i}/{dict_id}_{ccfg.dict_size}/ae.pt')))
        dictionaries[mlps[i]] = ae

        ae = AutoEncoder(args.d_model, ccfg.dict_size).to(args.device)
        ae.load_state_dict(t.load(os.path.join(args.dict_path, f'resid_out_layer{i}/{dict_id}_{ccfg.dict_size}/ae.pt')))
        dictionaries[resids[i]] = ae
