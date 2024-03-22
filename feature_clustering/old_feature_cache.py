# Imports and constants
import os
import torch
import torch as t
from nnsight import LanguageModel
from tqdm import trange
import json
import datasets
import gc
import einops

import sys
sys.path.append('/home/can/dictionary-circuits/')
from loading_utils import load_submodules_and_dictionaries_from_generic, DictionaryCfg
from cluster_utils import ClusterConfig, get_tokenized_context_y

# Set enviroment specific constants
dense_activations = False # If true, caching dense activations and gradients from model component output, else caching sparse dictionaries activations and gradients
batch_size = 16
results_dir = "/home/can/feature_clustering/clustering_pythia-70m-deduped_tloss0.1_nsamples16384_npos128_filtered-induction_attn-mlp-resid"
dictionary_dir="/share/projects/dictionary_circuits/autoencoders/pythia-70m-deduped"
tokenized_dataset_dir = "/home/can/data/pile_test_tokenized_600k/"
device = "cuda:0"
data_type = torch.float16

# Load config, data, model, dictionaries
ccfg = ClusterConfig(**json.load(open(os.path.join(results_dir, "config.json"), "r")))
final_token_idxs = torch.load(os.path.join(results_dir, f"final_token_idxs-tloss{ccfg.loss_threshold}-nsamples{ccfg.num_samples}-nctx{ccfg.n_pos}.pt"))
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
n_bytes_per_nonzero = 2 # using float32: 4 bytes per non-zero value
n_nonzero_per_sample = ccfg.n_pos * ccfg.dictionary_size * ccfg.n_submodules # upper bound if dense activations
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
## Sparse format on cpu, using COO bc it allows concatenation
# activation_results_cpu = t.sparse_coo_tensor(
#     size=(0, ccfg.n_pos * ccfg.dictionary_size * ccfg.n_submodules), 
#     dtype=torch.float32, 
#     device=device
#     )
# lin_effect_results_cpu = t.sparse_coo_tensor(
#     size=(0, ccfg.n_pos * ccfg.dictionary_size * ccfg.n_submodules), 
#     dtype=torch.float32, 
#     device=device
    # )

activations_per_batch = dict()
lin_effects_per_batch = dict()

for batch_idx in trange(n_batches, desc="Caching activations in batches", total=n_batches):
    torch.cuda.empty_cache()
    gc.collect()
    print(f"GPU Allocated memory: {torch.cuda.memory_allocated(device)/1024**2 :.2f} MB")
    print(f"GPU Cached memory: {torch.cuda.memory_reserved(device)/1024**2 :.2f} MB")

    contexts, ys = next(loader)
    contexts, ys = contexts.to(device), ys.to(device)
    activations = t.zeros((ccfg.n_submodules, batch_size, ccfg.n_pos, ccfg.dictionary_size), dtype=data_type, device=device)
    gradients = t.zeros((ccfg.n_submodules, batch_size, ccfg.n_pos, ccfg.dictionary_size), dtype=data_type, device=device)

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


#     activations = einops.rearrange(activations, 'n_submodules batch_size n_pos dictionary_size -> batch_size (n_pos n_submodules dictionary_size)')
#     gradients = einops.rearrange(gradients, 'n_submodules batch_size n_pos dictionary_size -> batch_size (n_pos n_submodules dictionary_size)')
    
#     # Calculate linear effects
#     lin_effects = (activations * gradients) # This elementwise mult would not be faster when setting activations to sparse, as gradients are dense.

#     ## Sparse format on cpu
#     activation_results_cpu = t.cat([activation_results_cpu, activations.to_sparse()], dim=0)
#     lin_effect_results_cpu = t.cat([lin_effect_results_cpu, lin_effects.to_sparse()], dim=0)

#     torch.cuda.empty_cache()
#     gc.collect()

# # Save results in sparse format
# t.save(activation_results_cpu, os.path.join(results_dir, f"activations.pt"))
# t.save(lin_effect_results_cpu, os.path.join(results_dir, f"lin_effects.pt"))
        
    activations = einops.rearrange(activations, 'n_submodules batch_size n_pos dictionary_size -> batch_size n_pos (n_submodules dictionary_size)')
    gradients = einops.rearrange(gradients, 'n_submodules batch_size n_pos dictionary_size -> batch_size n_pos (n_submodules dictionary_size)')
    
    # Calculate linear effects
    lin_effects = (activations * gradients) # This elementwise mult would not be faster when setting activations to sparse, as gradients are dense.

    # Final position
    activations = activations[:, -1, :].squeeze()
    lin_effects = lin_effects[:, -1, :].squeeze()
    print(f"nonzero activations: {t.nonzero(activations).shape[0]}")

    ## Sparse format on cpu
    activations_per_batch[batch_idx] = activations.to_sparse()
    lin_effects_per_batch[batch_idx] = lin_effects.to_sparse()

    # activations_memory = activations.element_size() * activations.nelement() / 1024**2
    # lin_effects_memory = lin_effects.element_size() * lin_effects.nelement() / 1024**2
    # print(f"act added to memory: {activations_memory} MB")
    # print(f"lin_effects added to memory: {lin_effects_memory} MB")

# Save results in sparse format (CSR is more storage efficient)
activation_results_cpu = t.cat([activations_per_batch[i] for i in activations_per_batch], dim=0)
lin_effect_results_cpu = t.cat([lin_effects_per_batch[i] for i in lin_effects_per_batch], dim=0)
t.save(activation_results_cpu, os.path.join(results_dir, f"activations.pt"))
t.save(lin_effect_results_cpu, os.path.join(results_dir, f"lin_effects.pt"))