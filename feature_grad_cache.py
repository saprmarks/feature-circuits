# %%
# Imports and constants

import os
import torch
import numpy as np
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from nnsight import LanguageModel
import datasets
from tqdm import tqdm
import json

import sys
sys.path.append('..')
from loading_utils import load_submodules_and_dictionaries_from_generic, DictionaryCfg, submodule_name_to_type

# Experiment parameters
model_name = "pythia-70m-deduped"
step = 143000
device = "cuda:0"
cache_dir = "/home/can/feature_clustering/cache/"
pile_canonical = "/home/can/data/pile_test_tokenized_200k/"
output_dir = "/home/can/feature_clustering/results/"
loss_threshold = 0.0001
num_tokens = int(1e3)
skip = 1
k = int(1e3)

# Submodule and dictionary parameters
submodules_generic = ['model.gpt_neox.layers.{}.mlp.dense_4h_to_h']
dict_cfg = DictionaryCfg(
    dictionary_size=512 * 64,
    dictionary_dir="/share/projects/dictionary_circuits/autoencoders/pythia-70m-deduped"
)
submod_type_names = "_".join([submodule_name_to_type(s) for s in submodules_generic])


# Approximate size of topk_feature_activations_per_token
key_size = 4 # bytes (int)
act_and_idx_size = 8 # bytes (float32)
topk_act_size = key_size + 2 * act_and_idx_size * k # Index, value, k times
total_size = 2 * num_tokens * (topk_act_size + key_size) # 2 times for activations and gradients
print(f"Total size of generated data (feature activations and gradients): {total_size / 1024**2} MB")


# %%
# Load model an dictionaries
model = LanguageModel("EleutherAI/"+model_name, device_map=device)

submodule_names, submodules, dictionaries = load_submodules_and_dictionaries_from_generic(
    model, submodules_generic, dict_cfg
)

# %%
# Load losses data
particular_model_cache_dir = os.path.join(cache_dir, model_name, f"step{step}")
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
# Load dataset and helper functions

# Dataset
dataset = datasets.load_from_disk(pile_canonical)

# Dataset handling
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

def print_context(idx):
    """
    given idx in range(0, 10658635), print prompt preceding the corresponding
    prediction, and highlight the predicted token.
    """
    sample, token_idx = get_context(idx)
    prompt = sample["split_by_token"][:token_idx]
    prompt = "".join(prompt)
    token = sample["split_by_token"][token_idx]
    print(prompt + "\033[41m" + token + "\033[0m")

def feat_idx_to_name(global_idx):
    """given a feature index, return a string containing layer, feat_index_in_submodule, submodule_type"""
    global_idx = int(global_idx)
    features_per_layer = len(submodules_generic) * dict_cfg.size # assuming all submodules have the same dictionary size
    layer_num = global_idx // features_per_layer
    feat_idx_layer = global_idx % features_per_layer
    submodule_num = feat_idx_layer // dict_cfg.size
    feat_idx_submodule = feat_idx_layer % dict_cfg.size
    submodule_type = submodule_name_to_type(submodules_generic[submodule_num])
    return f"{layer_num}_{feat_idx_submodule}_{submodule_type}"


# %%
# Metric
def metric_fn(model, targets, target_token_pos): # can't you do that with logits[:, target_token_idx, :]?
    logits = model.embed_out.output
    # m = torch.nn.functional.cross_entropy(logits[0, :-1, :], targets[0, 1:], reduction='none')[target_token_pos-1]
    m = -1 * torch.log_softmax(logits[0, :-1, :], dim=-1)
    target_token_id = targets[0, target_token_pos]
    return m[target_token_pos-1, target_token_id]


# %%
# Cache feature activations and gradients
topk_feature_activations_per_token = {}
topk_feature_gradients_per_token = {}

for token_loss_idx in tqdm(token_loss_idxs, desc="context", total=num_tokens):
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
    feature_activation_vector = torch.cat([v.value[0, target_token_pos-1] for v in activations.values()])
    feature_gradient_vector = torch.cat([v.value[0, target_token_pos-1] for v in gradients.values()])
    topk_activation_feat_idxs = torch.argsort(feature_activation_vector, descending=True)[:k]
    topk_gradient_feat_idxs = torch.argsort(feature_gradient_vector, descending=True)[:k]
    topk_feature_activations_per_token[token_loss_idx] = torch.stack((topk_activation_feat_idxs, feature_activation_vector[topk_activation_feat_idxs]), dim=-1).tolist() # Idx are stored as floats in torch tensors
    topk_feature_gradients_per_token[token_loss_idx] = torch.stack((topk_gradient_feat_idxs, feature_gradient_vector[topk_gradient_feat_idxs]), dim=-1).tolist() # Idx are stored as floats in torch tensors

# %%
# Save results
resultname = f"{model_name}_{loss_threshold}_{skip}_{num_tokens}_{submod_type_names}.json"
json.dump(topk_feature_activations_per_token, open(os.path.join(output_dir, f"act_{resultname}"), "w"))
json.dump(topk_feature_gradients_per_token, open(os.path.join(output_dir, f"grad_{resultname}"), "w"))