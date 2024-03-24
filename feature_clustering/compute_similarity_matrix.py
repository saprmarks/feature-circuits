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
from cluster_utils import ClusterConfig, get_tokenized_context_y, row_filter
from attribution import patching_effect
from dictionary_learning import AutoEncoder
import argparse


def data_loader(final_token_idxs, n_batches, batch_size, device):
    contexts, ys = t.zeros((n_batches, batch_size, ccfg.n_ctx)), t.zeros((n_batches, batch_size, 1), dtype=t.int, device=device)
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

# Cache feature activations and gradients and compute similarity blockwise
def cache_activations_and_effects(contexts, answers, model, submodules, dictionaries, batch_size, aggregation, device):
    def metric_fn(model):
        return (
            -1 * t.gather(
                t.nn.functional.log_softmax(model.embed_out.output[:,-1,:], dim=-1), dim=-1, index=answers.view(-1, 1)
            ).squeeze(-1)
        )

    effect_out = patching_effect(
        clean=contexts,
        patch=None,
        model=model,
        submodules=submodules,
        dictionaries=dictionaries,
        metric_fn=metric_fn,
        method="all-folded",
    )
    activations = t.zeros((ccfg.n_submodules, batch_size, ccfg.n_ctx, ccfg.dict_size), dtype=t.float32, device=device)
    effects = t.zeros((ccfg.n_submodules, batch_size, ccfg.n_ctx, ccfg.dict_size), dtype=t.float32, device=device)
    for i, submod in enumerate(submodules):
        activations[i] = effect_out.deltas[submod]
        effects[i] = effect_out.effects[submod]

    # Aggregation of positional information
    if aggregation == "sum":
        activations = einops.rearrange(activations, 'n_submodules batch_size n_pos dictionary_size -> batch_size n_pos (n_submodules dictionary_size)')
        effects = einops.rearrange(effects, 'n_submodules batch_size n_pos dictionary_size -> batch_size n_pos (n_submodules dictionary_size)')
        activations = activations.sum(dim=1)
        effects = effects.sum(dim=1)
    elif isinstance(aggregation, int):
        if aggregation < 0 or aggregation >= ccfg.n_ctx:
            raise ValueError("Aggregation value must be between 0 and n_ctx.")
        activations = einops.rearrange(activations, 'n_submodules batch_size n_pos dictionary_size -> batch_size n_pos (n_submodules dictionary_size)')
        effects = einops.rearrange(effects, 'n_submodules batch_size n_pos dictionary_size -> batch_size n_pos (n_submodules dictionary_size)')
        activations = activations[:, -aggregation:]
        effects = effects[:, -aggregation:]
    elif aggregation is None:
        activations = einops.rearrange(activations, 'n_submodules batch_size n_pos dictionary_size -> batch_size (n_pos n_submodules dictionary_size)')
        effects = einops.rearrange(effects, 'n_submodules batch_size n_pos dictionary_size -> batch_size (n_pos n_submodules dictionary_size)')
    else:
        raise ValueError("Invalid aggregation description.")
    return activations, effects

def compute_similarity_matrix(X1, X2, absolute=False):
    X1_norm = F.normalize(X1, p=2, dim=-1)
    X2_norm = F.normalize(X2, p=2, dim=-1)
    C = t.matmul(X1_norm, X2_norm.T) # cos similarity
    if absolute:
        return t.abs(C)
    return C

def compute_similarity_matrix_blockwise(contexts, answers, model, submodules, dictionaries, batch_size, aggregation, block_length, device, absolute=False):
    torch.cuda.empty_cache()
    gc.collect()

    C_act = t.zeros(ccfg.n_samples, ccfg.n_samples, device=device, dtype=t.float32)
    C_lin = t.zeros_like(C_act)
    n_blocks = ccfg.n_samples // block_length
    for i in trange(n_blocks, desc="Computing similarity blockwise, ROWS"):
        for j in trange(n_blocks, desc="COLS"):
            
            print(f"GPU Allocated memory: {torch.cuda.memory_allocated(device)/1024**2 :.2f} MB")
            print(f"GPU Cached memory: {torch.cuda.memory_reserved(device)/1024**2 :.2f} MB")
            X_row_act, X_row_lin = cache_activations_and_effects(contexts[i], answers[i], model, submodules, dictionaries, batch_size, aggregation, device)
            X_col_act, X_col_lin = cache_activations_and_effects(contexts[j], answers[j], model, submodules, dictionaries, batch_size, aggregation, device)
            C_block_act = compute_similarity_matrix(X_row_act, X_col_act, absolute=absolute)
            C_block_lin = compute_similarity_matrix(X_row_lin, X_col_lin, absolute=absolute)
            C_act[i*block_length:(i+1)*block_length, j*block_length:(j+1)*block_length] = C_block_act
            C_lin[i*block_length:(i+1)*block_length, j*block_length:(j+1)*block_length] = C_block_lin
            if i != j: # fill the transpose of the block if not on diagonal
                C_act[j*block_length:(j+1)*block_length, i*block_length:(i+1)*block_length] = C_block_act.T
                C_lin[j*block_length:(j+1)*block_length, i*block_length:(i+1)*block_length] = C_block_lin.T
    return C_act, C_lin


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute similarity matrices of feature activations and linear effects")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for data loader")
    parser.add_argument("--aggregation", type=str, default=None, help="Method for aggregating positional information ('sum' or int x for final x positions)")
    parser.add_argument("--results_dir", type=str, default="/home/can/dictionary-circuits/feature_clustering/clusters/dataset_nsamples32768_nctx16_tloss0.1_filtered-induction_attn-mlp-resid_pythia-70m-deduped", help="Directory to load data and save results")
    parser.add_argument("--dict_path", type=str, default="/share/projects/dictionary_circuits/autoencoders/pythia-70m-deduped", help="Directory of dictionaries")
    parser.add_argument("--tokenized_dataset_dir", type=str, default="/home/can/data/pile_test_tokenized_600k/", help="Directory of tokenized dataset")
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--dict_id", type=int, default=10, help="Dictionary id")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for computation")
    args = parser.parse_args()


    # Load config, data, model
    ccfg = ClusterConfig(**json.load(open(os.path.join(args.results_dir, "config.json"), "r")))
    model = LanguageModel("EleutherAI/"+ccfg.model_name, device_map=args.device)

    embed = model.gpt_neox.embed_in
    attns = [layer.attention for layer in model.gpt_neox.layers]
    mlps = [layer.mlp for layer in model.gpt_neox.layers]
    resids = [layer for layer in model.gpt_neox.layers]
    submodules = [embed] + attns + mlps + resids
    # save n_submodules to the config
    # ccfg.n_submodules = 3 * model.config.num_hidden_layers + 1
    # with open(os.path.join(ccfg.results_dir, "config.json"), "w") as f:
    #     json.dump(ccfg.__dict__, f)

    dictionaries = {}
    if args.dict_id == 'id':
        from dictionary_learning.dictionary import IdentityDict
        dictionaries[embed] = IdentityDict(args.d_model)
        for i in range(len(model.gpt_neox.layers)):
            dictionaries[attns[i]] = IdentityDict(args.d_model)
            dictionaries[mlps[i]] = IdentityDict(args.d_model)
            dictionaries[resids[i]] = IdentityDict(args.d_model)
    else:
        for i in range(len(model.gpt_neox.layers)):
            ae = AutoEncoder(args.d_model, ccfg.dict_size).to(args.device)
            ae.load_state_dict(t.load(os.path.join(args.dict_path, f'embed/{args.dict_id}_{ccfg.dict_size}/ae.pt')))
            dictionaries[embed] = ae

            ae = AutoEncoder(args.d_model, ccfg.dict_size).to(args.device)
            ae.load_state_dict(t.load(os.path.join(args.dict_path, f'attn_out_layer{i}/{args.dict_id}_{ccfg.dict_size}/ae.pt')))
            dictionaries[attns[i]] = ae

            ae = AutoEncoder(args.d_model, ccfg.dict_size).to(args.device)
            ae.load_state_dict(t.load(os.path.join(args.dict_path, f'mlp_out_layer{i}/{args.dict_id}_{ccfg.dict_size}/ae.pt')))
            dictionaries[mlps[i]] = ae

            ae = AutoEncoder(args.d_model, ccfg.dict_size).to(args.device)
            ae.load_state_dict(t.load(os.path.join(args.dict_path, f'resid_out_layer{i}/{args.dict_id}_{ccfg.dict_size}/ae.pt')))
            dictionaries[resids[i]] = ae

    # Load and prepare data
    final_token_idxs = torch.load(os.path.join(args.results_dir, f"final_token_idxs-tloss{ccfg.loss_threshold}-nsamples{ccfg.n_samples}-nctx{ccfg.n_ctx}.pt"))
    dataset = datasets.load_from_disk(args.tokenized_dataset_dir)
    n_batches = ccfg.n_samples // args.batch_size # Not achieving the exact number of samples if n_samples is not divisible by args.batch_size
    contexts, answers = data_loader(final_token_idxs, n_batches, args.batch_size, args.device)

    # Compute similarity matrices
    act_similarity_matrix, lin_similarity_matrix = compute_similarity_matrix_blockwise(
        contexts,
        answers,
        model,
        submodules,
        dictionaries,
        args.batch_size,
        args.aggregation,
        ccfg.n_ctx,
        args.device,
        absolute=False
    )

    # Save
    act_similarity_matrix_filename = f"similarity_matrix_activations_nsamples{ccfg.n_samples}_nctx{ccfg.n_ctx}_{args.aggregation}_dict{args.dict_id}.pt"
    lin_similarity_matrix_filename = f"similarity_matrix_lin_effects_nsamples{ccfg.n_samples}_nctx{ccfg.n_ctx}_{args.aggregation}_dict{args.dict_id}.pt"
    t.save(act_similarity_matrix, os.path.join(ccfg.results_dir, act_similarity_matrix_filename))
    t.save(act_similarity_matrix, os.path.join(ccfg.results_dir, act_similarity_matrix_filename))


# %%
# test for comparing functions compute_similarity_matrix_blockwise and compute_similarity_matrix
# X = t.randn(100, 1024)
# C_blockwise = compute_similarity_matrix_blockwise(X, angular=False, block_length=4, device="cpu")
# C = compute_similarity_matrix(X, X, angular=False)
# assert t.allclose(C_blockwise, C, atol=1e-5)