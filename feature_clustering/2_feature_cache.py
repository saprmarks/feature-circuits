# Imports and constants
import os
import torch
import torch as t
from nnsight import LanguageModel
from tqdm import trange
import json
import datasets
from collections import defaultdict
import gc
import einops

import sys
sys.path.append('/home/can/dictionary-circuits/')
from loading_utils import load_submodules_and_dictionaries_from_generic, DictionaryCfg, submodule_name_to_type
from cluster_utils import ClusterConfig, get_tokenized_context_y

# Set enviroment specific constants
batch_size = 16
results_dir = "/home/can/feature_clustering/clustering_pythia-70m-deduped_tloss0.1_nsamples8192_npos64_filtered-induction_mlp-attn-resid"
dictionary_dir="/share/projects/dictionary_circuits/autoencoders/pythia-70m-deduped"
tokenized_dataset_dir = "/home/can/data/pile_test_tokenized_600k/"
device = "cuda:0"

# Load config, data, model, dictionaries
ccfg = ClusterConfig(**json.load(open(os.path.join(results_dir, "config.json"), "r")))
final_token_idxs = torch.load(os.path.join(results_dir, "final_token_idxs.pt"))
dataset = datasets.load_from_disk(tokenized_dataset_dir)
model = LanguageModel("EleutherAI/"+ccfg.model_name, device_map=device)
dict_cfg = DictionaryCfg(dictionary_size=ccfg.dictionary_size, dictionary_dir=dictionary_dir)
submodule_names, submodules, dictionaries = load_submodules_and_dictionaries_from_generic(model, ccfg.submodules_generic, dict_cfg)
n_batches = ccfg.num_samples // batch_size # Not achieving the exact number of samples if num_samples is not divisible by batch_size
ccfg.n_submodules = len(submodule_names[0]) * model.config.num_hidden_layers
# save n_submodules to the config
with open(os.path.join(results_dir, "config.json"), "w") as f:
    json.dump(ccfg.__dict__, f)

# Approximate size of activation_results_cpu and lin_effect_results_cpu 
# Sparse activations
n_bytes_per_nonzero = 4 # using float32: 4 bytes per non-zero value
n_nonzero_per_sample = ccfg.n_pos * ccfg.dictionary_size * ccfg.n_submodules
total_size = n_nonzero_per_sample * n_batches * batch_size * n_bytes_per_nonzero
print(f"Total size of generated data (feature activations): {total_size / 1024**3} GB")

# Data loader
def data_loader(final_token_idxs, batch_size):
    for i in range(n_batches):
        contexts, ys = t.zeros((batch_size, ccfg.n_pos)), t.zeros((batch_size, 1))
        for j in range(batch_size):
            sample_idx = i * batch_size + j
            context, y, _ = get_tokenized_context_y(
                ccfg, 
                dataset, 
                doc_idx=int(final_token_idxs[sample_idx, 0]), 
                final_token_in_context_index=int(final_token_idxs[sample_idx, 1])
                )
            contexts[j] = t.tensor(context)
            ys[j] = t.tensor(y)
        yield contexts.int(), ys.int()
loader = data_loader(final_token_idxs, batch_size)

# Metric
def metric_fn(logits, target_token_id): # logits shape: (batch_size, seq_len, vocab_size)
    m = torch.log_softmax(logits[:, -1, :], dim=-1) # batch_size, vocab_size
    m = m[t.arange(m.shape[0]), target_token_id] # batch_size
    return m.sum()

# Cache feature activations and gradients

## Dense format on cpu
# activation_results_cpu = t.zeros((n_batches * batch_size, ccfg.n_pos, ccfg.dictionary_size * n_submodules), dtype=torch.float32, device='cpu')
# lin_effect_results_cpu = t.zeros((n_batches * batch_size, ccfg.n_pos, ccfg.dictionary_size * n_submodules), dtype=torch.float32, device='cpu')

## Sparse format on cpu
activation_results_cpu = t.sparse_coo_tensor(
    size=(0, ccfg.n_pos, ccfg.dictionary_size * ccfg.n_submodules), 
    dtype=torch.float32, 
    device='cpu'
    )
lin_effect_results_cpu = t.sparse_coo_tensor(
    size=(0, ccfg.n_pos, ccfg.dictionary_size * ccfg.n_submodules), 
    dtype=torch.float32, 
    device='cpu'
    )

# activation_results_batch_idxs = t.tensor([])
# activation_results_pos_idxs = t.tensor([])
# activation_results_feat_idxs = t.tensor([])
# activation_results_values = t.tensor([])

# lin_effect_results_batch_idxs = t.tensor([])
# lin_effect_results_pos_idxs = t.tensor([])
# lin_effect_results_feat_idxs = t.tensor([])
# lin_effect_results_values = t.tensor([])

for batch_idx in trange(n_batches, desc="Caching activations in batches", total=n_batches):
    print(f"GPU Allocated memory: {torch.cuda.memory_allocated(device)/1024**2 :.2f} MB")
    print(f"GPU Cached memory: {torch.cuda.memory_reserved(device)/1024**2 :.2f} MB")

    contexts, ys = next(loader)
    contexts, ys = contexts.to(device), ys.to(device)
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
    
    # # Flatten activations and gradients
    # activations_flat = torch.zeros((batch_size, ccfg.n_pos, ccfg.dictionary_size * ccfg.n_submodules), dtype=torch.float32, device=device)
    # gradients_flat = torch.zeros((batch_size, ccfg.n_pos, ccfg.dictionary_size * ccfg.n_submodules), dtype=torch.float32, device=device)
    # for i, name in enumerate(activations.keys()):
    #     # Calculate the start and end indices for the current submodule in the flat tensor, assign to flat tensors
    #     feature_start_idx = i * ccfg.dictionary_size
    #     feature_end_idx = feature_start_idx + ccfg.dictionary_size
    #     activations_flat[:, :, feature_start_idx:feature_end_idx] = activations[name]
    #     gradients_flat[:, :, feature_start_idx:feature_end_idx] = gradients[name]
    
    # Calculate linear effects
    lin_effects = (activations * gradients) # This elementwise mult would not be faster when setting activations to sparse, as gradients are dense.

    # Move to cpu
    ## Dense format on cpu
    # batch_start_idx = batch_idx * batch_size
    # results_end_idx = batch_start_idx + batch_size
    # activation_results_cpu[batch_start_idx:results_end_idx] = activations.to_sparse().cpu()
    # lin_effect_results_cpu[batch_start_idx:results_end_idx] = lin_effects.to_sparse().cpu()

    ## Sparse format on cpu, t.cat is bugged.
    activation_results_cpu = t.cat([activation_results_cpu, activations.to_sparse().cpu()], dim=0)
    lin_effect_results_cpu = t.cat([lin_effect_results_cpu, lin_effects.to_sparse().cpu()], dim=0)

    # activation_sparse = activations.to_sparse().cpu()
    # lin_effect_sparse = lin_effects.to_sparse().cpu()

    # # Save indices manually, convert to sparse format later
    # batch_idxs = activation_sparse.indices()[0] + batch_idx * batch_size
    # activation_results_batch_idxs = t.hstack((activation_results_batch_idxs, batch_idxs))
    # activation_results_pos_idxs = t.hstack((activation_results_pos_idxs, activation_sparse.indices()[1]))
    # activation_results_feat_idxs = t.hstack((activation_results_feat_idxs, activation_sparse.indices()[2]))
    # activation_results_values = t.hstack((activation_results_values, activation_sparse.values()))
    # current_activation_results_all_indices = t.vstack((activation_results_batch_idxs, activation_results_pos_idxs, activation_results_feat_idxs))
    # current_activation_results = t.sparse_coo_tensor(current_activation_results_all_indices, activation_results_values, size=(batch_size * (batch_idx+1), ccfg.n_pos, ccfg.dictionary_size * ccfg.n_submodules), dtype=torch.float32, device='cpu')
    
    # batch_idxs = lin_effect_sparse.indices()[0] + batch_idx * batch_size # redundant, but for clarity
    # lin_effect_results_batch_idxs = t.hstack((lin_effect_results_batch_idxs, batch_idxs))
    # lin_effect_results_pos_idxs = t.hstack((lin_effect_results_pos_idxs, lin_effect_sparse.indices()[1]))
    # lin_effect_results_feat_idxs = t.hstack((lin_effect_results_feat_idxs, lin_effect_sparse.indices()[2]))
    # lin_effect_results_values = t.hstack((lin_effect_results_values, lin_effect_sparse.values()))

    torch.cuda.empty_cache()
    gc.collect()

# Save results in sparse format
## Dense format on cpu
# activation_results_cpu = activation_results_cpu.to_sparse()
# lin_effect_results_cpu = lin_effect_results_cpu.to_sparse()
# activation_results_all_indices = t.vstack((activation_results_batch_idxs, activation_results_pos_idxs, activation_results_feat_idxs))
# activation_results_sparse = t.sparse_coo_tensor(activation_results_all_indices, activation_results_values, size=(n_batches * batch_size, ccfg.n_pos, ccfg.dictionary_size * ccfg.n_submodules), dtype=torch.float32, device='cpu')

# lin_effect_results_all_indices = t.vstack((lin_effect_results_batch_idxs, lin_effect_results_pos_idxs, lin_effect_results_feat_idxs))
# lin_effect_results_sparse = t.sparse_coo_tensor(lin_effect_results_all_indices, lin_effect_results_values, size=(n_batches * batch_size, ccfg.n_pos, ccfg.dictionary_size * ccfg.n_submodules), dtype=torch.float32, device='cpu')

t.save(activation_results_cpu, os.path.join(results_dir, f"activations.pt"))
t.save(lin_effect_results_cpu, os.path.join(results_dir, f"lin_effects.pt"))