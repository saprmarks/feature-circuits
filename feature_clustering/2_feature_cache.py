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

import sys
sys.path.append('/home/can/dictionary-circuits/')
from loading_utils import load_submodules_and_dictionaries_from_generic, DictionaryCfg, submodule_name_to_type
from cluster_utils import ClusterConfig, get_tokenized_context_y

# Set enviroment specific constants
device = "cuda:0"
batch_size = 16
results_dir = "/home/can/feature_clustering/clustering_pythia-70m-deduped_tloss0.03_ntok16384_skip256_npos256_mlp"
dictionary_dir="/share/projects/dictionary_circuits/autoencoders/pythia-70m-deduped"
tokenized_dataset_dir = "/home/can/data/pile_test_tokenized_600k/"

# Load config, data, model, dictionaries
ccfg = ClusterConfig(**json.load(open(os.path.join(results_dir, "config.json"), "r")))
final_token_idxs = torch.load(os.path.join(results_dir, "final_token_idxs.pt"))
dataset = datasets.load_from_disk(tokenized_dataset_dir)
model = LanguageModel("EleutherAI/"+ccfg.model_name, device_map=device)
dict_cfg = DictionaryCfg(dictionary_size=ccfg.dictionary_size, dictionary_dir=dictionary_dir)
submodule_names, submodules, dictionaries = load_submodules_and_dictionaries_from_generic(model, ccfg.submodules_generic, dict_cfg)
n_batches = ccfg.num_samples // batch_size

# Approximate size of topk_feature_activations_per_token
# results will be saved as dict d where d[token_loss_idx] = {"sum": [[feat_idx, act, grad], ...], "pos": {pos_idx(negative): [[feat_idx, act, grad], ...], ...}
pos_size = (4 + 4 + 4) * 300 * ccfg.n_pos # 300 is an estimate for the number of nonzero feature activations per position
sum_size = (4 + 4 + 4) * 15000 # is an estimate for the number of nonzero feature activations across all positions
total_size = (pos_size + sum_size) * ccfg.num_samples
print(f"Total size of generated data (feature activations): {total_size / 1024**3} GB")

# Data loader
def data_loader(final_token_idxs, batch_size):
    for i in range(n_batches):
        contexts, ys = t.zeros((batch_size, ccfg.n_pos)), t.zeros((batch_size, 1))
        for j in range(i, i+batch_size):
            context, y, _ = get_tokenized_context_y(
                ccfg, 
                dataset, 
                doc_idx=int(final_token_idxs[j, 0]), 
                final_token_in_context_index=int(final_token_idxs[j, 1])
                )
            contexts[j-i] = t.tensor(context)
            ys[j-i] = t.tensor(y)
        yield contexts.int(), ys.int()
loader = data_loader(final_token_idxs, batch_size)

# Metric
def metric_fn(logits, target_token_id): # logits shape: (batch_size, seq_len, vocab_size)
    m = torch.log_softmax(logits[:, -1, :], dim=-1) # batch_size, vocab_size
    m = m[t.arange(m.shape[0]), target_token_id] # batch_size
    return m.sum()

# Cache feature activations and gradients
for batch_idx in trange(n_batches, desc="Caching activations", total=n_batches):
    contexts, ys = next(loader) # for batch_size=1
    contexts, ys = contexts.to(device), ys.to(device)

    print(f"Allocated memory: {torch.cuda.memory_allocated(device)/1024**2 :.2f} MB")
    print(f"Cached memory: {torch.cuda.memory_reserved(device)/1024**2 :.2f} MB")

    activations = defaultdict(list)
    gradients = defaultdict(list)


    with model.invoke(contexts, fwd_args={'inference': False}) as invoker:
        for layer in range(model.config.num_hidden_layers):
            for name, sm, ae in zip(submodule_names[layer], submodules[layer], dictionaries[layer]):
                x = sm.output
                is_resid = (type(x.shape) == tuple)
                if is_resid:
                    x = x[0]
                f = ae.encode(x)
                activations[name] = f.detach().save()
                gradients[name] = f.grad.detach().save() # [batch_size, seq_len, vocab_size]

                
                x_hat = ae.decode(f)
                residual = (x - x_hat).detach()
                if is_resid:
                    sm.output[0][:] = x_hat + residual
                else:
                    sm.output = x_hat + residual
        logits = model.embed_out.output # [batch_size, seq_len, vocab_size]
        metric_fn(logits=logits, target_token_id=ys).backward()
    
    # Flatten activations and gradients
    n_submodules = len(submodule_names[0]) * model.config.num_hidden_layers
    activations_flat = torch.zeros((n_submodules, batch_size, ccfg.n_pos, ccfg.dictionary_size), dtype=torch.float32, device=device)
    gradients_flat = torch.zeros((n_submodules, batch_size, ccfg.n_pos, ccfg.dictionary_size), dtype=torch.float32, device=device)
    for i, name in enumerate(activations.keys()):
        activations_flat[i] = activations[name]
        gradients_flat[i] = gradients[name]
    
    # Calculate linear effects
    lin_effects = (activations_flat * gradients_flat) # This elementwise mult would not be faster when setting activations to sparse, as gradients are dense.
    activations_flat.to_sparse()
    lin_effects.to_sparse()

    # Save results
    t.save(activations_flat, os.path.join(results_dir, f"activations_batch{batch_idx}of{n_batches}.pt"))
    t.save(lin_effects, os.path.join(results_dir, f"lin_effects_batch{batch_idx}of{n_batches}.pt"))

    del activations_flat
    del gradients_flat
    del lin_effects
    torch.cuda.empty_cache()
    gc.collect()




 # Conversion and saving activations currently only works for batch_size=1
    # Convert to 1D feature vector
    # sum_feature_activation_vector = torch.cat([v.value[0].sum(dim=0) for v in activations.values()])
    # sum_feature_gradient_vector = torch.cat([v.value[0].sum(dim=0) for v in gradients.values()])
    # sum_activation_feat_idxs = torch.nonzero(sum_feature_activation_vector).flatten() # using this index for both activations and gradients
    # print(f'\nnumber of nonzero feature activations in sum: {len(sum_activation_feat_idxs)}')
    
    # sum_stack = torch.stack((
    #     sum_activation_feat_idxs.int(), # feature idx
    #     sum_feature_activation_vector[sum_activation_feat_idxs], 
    #     sum_feature_gradient_vector[sum_activation_feat_idxs]),
    #     dim=-1).cpu().tolist() # Idx are stored as floats in torch tensors


# pos_feature_activation_vector = torch.cat([v.value[0][-n_pos:] for v in activations.values()], dim=1) # shape (n_pos, n_features)
    # pos_feature_gradient_vector = torch.cat([v.value[0][:-n_pos] for v in gradients.values()], dim=1)
    # pos_activation_feat_idxs = torch.nonzero(pos_feature_activation_vector)
    # pos_stack = torch.stack((
    #     pos_activation_feat_idxs[:, 0].int()-n_pos, # position idx, converted to final indices
    #     pos_activation_feat_idxs[:, 1].int(), # feature idx
    #     pos_feature_activation_vector[pos_activation_feat_idxs[:, 0], pos_activation_feat_idxs[:, 1]],
    #     pos_feature_gradient_vector[pos_activation_feat_idxs[:, 0], pos_activation_feat_idxs[:, 1]]),
    #     dim=-1).cpu().tolist() # Idx are stored as floats in torch tensors

    # results[token_loss_idxs] = dict(sum=sum_stack, pos=pos_stack)

    # if i % 10 == 4:
    #     print(f"Allocated memory: {torch.cuda.memory_allocated(device)/1024**2 :.2f} MB")
    #     print(f"Cached memory: {torch.cuda.memory_reserved(device)/1024**2 :.2f} MB")