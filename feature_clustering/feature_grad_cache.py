# %%
# Imports and constants
import os
import torch
import numpy as np
from nnsight import LanguageModel
import datasets
from tqdm import tqdm
import json
import sys
sys.path.append('/home/can/dictionary-circuits/')
from loading_utils import load_submodules_and_dictionaries_from_generic, DictionaryCfg, submodule_name_to_type


# Experiment parameters
model_name = "pythia-70m-deduped"
device = "cuda:0"
cache_dir = "/home/can/feature_clustering/model_cache/"
pile_canonical = "/home/can/data/pile_test_tokenized_600k/"
activations_dir = "/home/can/feature_clustering/activations/"

start_token_idx = 4000

loss_threshold = 0.03
num_tokens = int(1e4)
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


# %%
# Approximate size of topk_feature_activations_per_token
# results will be saved as dict d where d[token_loss_idx] = {"sum": [[feat_idx, act, grad], ...], "pos": {-1: [feat_idx, act, grad], ...], -2: ...}
pos_size = (4 + 4 + 4) * 300 * n_pos 
sum_size = (4 + 4 + 4) * 1500
total_size = (pos_size + sum_size) * num_tokens
print(f"Total size of generated data (feature activations): {total_size / 1024**3} GB")


# %%
# Load losses data
particular_model_cache_dir = os.path.join(cache_dir, model_name)
losses_cached = [f for f in os.listdir(particular_model_cache_dir) if f.endswith("losses.pt")]
max_i = max(list(range(len(losses_cached))), key=lambda i: int(losses_cached[i].split("_")[0]))
docs, tokens = int(losses_cached[max_i].split("_")[0]), int(losses_cached[max_i].split("_")[2])
losses = torch.load(os.path.join(particular_model_cache_dir, f"{docs}_docs_{tokens}_tokens_losses.pt"))
c = 1 / np.log(2) # for nats to bits conversion

token_loss_idxs = (losses < (loss_threshold / c)).nonzero().flatten()
token_loss_idxs = token_loss_idxs[::skip]
token_loss_idxs = token_loss_idxs[:num_tokens].tolist()
assert len(token_loss_idxs) == num_tokens, "not enough tokens to analyze"


# %%
# Load model an dictionaries
model = LanguageModel("EleutherAI/"+model_name, device_map=device)
submodule_names, submodules, dictionaries = load_submodules_and_dictionaries_from_generic(
    model, submodules_generic, dict_cfg
)


# %%
# Load dataset and helper functions
dataset = datasets.load_from_disk(pile_canonical)
starting_indexes = np.array([0] + list(np.cumsum(dataset["preds_len"])))

def loss_idx_to_dataset_idx(idx):
    """given an idx in range(0, 10658635), return
    a sample index in range(0, 20000) and pred-in-sample
    index in range(0, 1023). Note token-in-sample idx is
    exactly pred-in-sample + 1"""
    sample_index = np.searchsorted(starting_indexes, idx, side="right") - 1
    pred_in_sample_index = idx - starting_indexes[sample_index]
    return int(sample_index), int(pred_in_sample_index)

def get_context(idx):
    """given idx in range(0, 10658635), return dataset sample
    and predicted token index within sample, in range(1, 1024)."""
    sample_index, pred_index = loss_idx_to_dataset_idx(idx)
    return dataset[sample_index], pred_index+1


# %%
# Metric
def metric_fn(model, targets, target_token_pos): # can't you do that with logits[:, target_token_idx, :]?
    logits = model.embed_out.output
    # m = torch.nn.functional.cross_entropy(logits[0, :-1, :], targets[0, 1:], reduction='none')[target_token_pos-1]
    m = torch.log_softmax(logits[0, :-1, :], dim=-1)
    target_token_id = targets[0, target_token_pos]
    return m[target_token_pos-1, target_token_id]


# %%
# Cache feature activations and gradients
results = dict()
enumerated = list(enumerate(token_loss_idxs))

for i, token_loss_idx in tqdm(enumerate(enumerated[start_token_idx:]), desc="context", total=num_tokens-start_token_idx):
    activations = {}
    gradients = {}
    doc, target_token_pos = get_context(token_loss_idx)
    token_ids = torch.tensor(doc['input_ids']).to(device)
    with model.invoke(token_ids, fwd_args={'inference': False}) as invoker:
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
        metric_fn(model, targets=token_ids, target_token_pos=target_token_pos).backward()


    # Convert to 1D feature vector
    sum_feature_activation_vector = torch.cat([v.value[0].sum(dim=0) for v in activations.values()])
    sum_feature_gradient_vector = torch.cat([v.value[0].sum(dim=0) for v in gradients.values()])
    sum_activation_feat_idxs = torch.nonzero(sum_feature_activation_vector).flatten() # using this index for both activations and gradients
    # print(f'\nnumber of nonzero feature activations in sum: {len(sum_activation_feat_idxs)}')
    sum_stack = torch.stack((
        sum_activation_feat_idxs.int(), # feature idx
        sum_feature_activation_vector[sum_activation_feat_idxs], 
        sum_feature_gradient_vector[sum_activation_feat_idxs]),
        dim=-1).tolist() # Idx are stored as floats in torch tensors
    
    pos_feature_activation_vector = torch.cat([v.value[0][-n_pos:] for v in activations.values()], dim=1) # shape (n_pos, n_features)
    pos_feature_gradient_vector = torch.cat([v.value[0][:-n_pos] for v in gradients.values()], dim=1)
    pos_activation_feat_idxs = torch.nonzero(pos_feature_activation_vector)
    pos_stack = torch.stack((
        pos_activation_feat_idxs[:, 0].int()-n_pos, # position idx, converted to final indices
        pos_activation_feat_idxs[:, 1].int(), # feature idx
        pos_feature_activation_vector[pos_activation_feat_idxs[:, 0], pos_activation_feat_idxs[:, 1]],
        pos_feature_gradient_vector[pos_activation_feat_idxs[:, 0], pos_activation_feat_idxs[:, 1]]),
        dim=-1).tolist() # Idx are stored as floats in torch tensors
    
    results[token_loss_idx] = dict(sum=sum_stack, pos=pos_stack)

    # Save results
    if i % 1000 == 999:
        json.dump(results, open(os.path.join(activations_dir, f"act-n-grad-{i}_{param_summary}.json"), "w"))
        torch.cuda.empty_cache()
        feature_act_grad_per_token = {}

json.dump(feature_act_grad_per_token, open(os.path.join(activations_dir, f"act-n-grad-last_{param_summary}.json"), "w"))