
# %%
# Imports and constants
import os
import torch
from nnsight import LanguageModel
from tqdm import trange
import json
import sys
sys.path.append('/home/can/dictionary-circuits/')
from loading_utils import load_submodules_and_dictionaries_from_generic, DictionaryCfg, submodule_name_to_type
from cluster_utils import LossesDataset

# with torch.autograd.profiler.profile(use_cuda=True) as prof:
# Experiment parameters
model_name = "pythia-70m-deduped"
device = "cuda:0"
model_cache_dir = "/home/can/feature_clustering/model_cache/"
tokenized_dataset_dir = "/home/can/data/pile_test_tokenized_600k/"
activations_dir = "/home/can/feature_clustering/activations/"

num_tokens = int(1e4)
batch_size = 1
n_batches = num_tokens // batch_size

start_token_idx = 0
save_every_n_batches = 100

loss_threshold = 0.03
skip = 512 # Trying to ramp this up to choose from diverse documents
n_pos = 10 # Saving feature activations for the final n_pos positions of each context

# Submodule and dictionary parameters
submodules_generic = ['model.gpt_neox.layers.{}.mlp.dense_4h_to_h']
dict_cfg = DictionaryCfg(
    dictionary_size=512 * 64,
    dictionary_dir="/share/projects/dictionary_circuits/autoencoders/pythia-70m-deduped"
)
submod_type_names = "-".join([submodule_name_to_type(s) for s in submodules_generic])
param_summary = f"{model_name}_tloss{loss_threshold}_ntok{num_tokens}_skip{skip}_npos{n_pos}_{submod_type_names}"


# Approximate size of topk_feature_activations_per_token
# results will be saved as dict d where d[token_loss_idx] = {"sum": [[feat_idx, act, grad], ...], "pos": {pos_idx(negative): [[feat_idx, act, grad], ...], ...}
pos_size = (4 + 4 + 4) * 300 * n_pos # 300 is an estimate for the number of nonzero feature activations per position
sum_size = (4 + 4 + 4) * 15000 # is an estimate for the number of nonzero feature activations across all positions
total_size = (pos_size + sum_size) * num_tokens
print(f"Total size of generated data (feature activations): {total_size / 1024**3} GB")


# Load model an dictionaries
model = LanguageModel("EleutherAI/"+model_name, device_map=device)
submodule_names, submodules, dictionaries = load_submodules_and_dictionaries_from_generic(
    model, submodules_generic, dict_cfg
)
generator_tokenized = LossesDataset(
    model,
    model_name, 
    model_cache_dir, 
    tokenized_dataset_dir, 
    loss_threshold, 
    num_tokens, 
    skip, 
    batch_size).generator()


#%%
# Metric
def metric_fn(model, target_token_ids): # adapted for batch_size=1
    logits = model.embed_out.output # shape (batch_size, seq_len, vocab_size)
    m = torch.log_softmax(logits[0, -1, :], dim=-1) # shape (vocab_size)
    return m[target_token_ids-1]


# Cache feature activations and gradients
results = dict()

for batch_idx in trange(n_batches, desc="context", total=num_tokens-start_token_idx):
    activations = {}
    gradients = {}
    contexts, ys, token_loss_idxs = next(generator_tokenized)
    contexts, ys, token_loss_idxs = contexts[0], ys[0], token_loss_idxs[0] # for batch_size=1
    # contexts, ys = contexts.to(device), ys.to(device) # not necessary for batch_size=1
    with model.invoke(contexts, fwd_args={'inference': False}) as invoker:
        for layer in range(model.config.num_hidden_layers):
            for name, sm, ae in zip(submodule_names[layer], submodules[layer], dictionaries[layer]):
                x = sm.output
                is_resid = (type(x.shape) == tuple)
                if is_resid:
                    x = x[0]
                f = ae.encode(x)
                activations[name] = f.detach().save()
                gradients[name] = f.grad.detach().save()
                
                x_hat = ae.decode(f)
                residual = (x - x_hat).detach()
                if is_resid:
                    sm.output[0][:] = x_hat + residual
                else:
                    sm.output = x_hat + residual   
        metric_fn(model, target_token_ids=ys).backward()

    # Conversion and saving activations currently only works for batch_size=1
    # Convert to 1D feature vector
    sum_feature_activation_vector = torch.cat([v.value[0].sum(dim=0) for v in activations.values()])
    sum_feature_gradient_vector = torch.cat([v.value[0].sum(dim=0) for v in gradients.values()])
    sum_activation_feat_idxs = torch.nonzero(sum_feature_activation_vector).flatten() # using this index for both activations and gradients
    # print(f'\nnumber of nonzero feature activations in sum: {len(sum_activation_feat_idxs)}')
    sum_stack = torch.stack((
        sum_activation_feat_idxs.int(), # feature idx
        sum_feature_activation_vector[sum_activation_feat_idxs], 
        sum_feature_gradient_vector[sum_activation_feat_idxs]),
        dim=-1).cpu().tolist() # Idx are stored as floats in torch tensors
    
    pos_feature_activation_vector = torch.cat([v.value[0][-n_pos:] for v in activations.values()], dim=1) # shape (n_pos, n_features)
    pos_feature_gradient_vector = torch.cat([v.value[0][:-n_pos] for v in gradients.values()], dim=1)
    pos_activation_feat_idxs = torch.nonzero(pos_feature_activation_vector)
    pos_stack = torch.stack((
        pos_activation_feat_idxs[:, 0].int()-n_pos, # position idx, converted to final indices
        pos_activation_feat_idxs[:, 1].int(), # feature idx
        pos_feature_activation_vector[pos_activation_feat_idxs[:, 0], pos_activation_feat_idxs[:, 1]],
        pos_feature_gradient_vector[pos_activation_feat_idxs[:, 0], pos_activation_feat_idxs[:, 1]]),
        dim=-1).cpu().tolist() # Idx are stored as floats in torch tensors
    
    torch.cuda.empty_cache()
    results[token_loss_idxs] = dict(sum=sum_stack, pos=pos_stack)

    # if i % 10 == 4:
    #     print(f"Allocated memory: {torch.cuda.memory_allocated(device)/1024**2 :.2f} MB")
    #     print(f"Cached memory: {torch.cuda.memory_reserved(device)/1024**2 :.2f} MB")

    # Save results
    if batch_idx % save_every_n_batches == save_every_n_batches-1:
        json.dump(results, open(os.path.join(activations_dir, f"act-n-grad-{batch_idx}_{param_summary}.json"), "w"))
        results = dict()

#%%
1e4 * 6 * 10 * 512 / 1024**3

# %%
