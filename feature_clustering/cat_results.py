# %%
# Imports and constants
import os
import torch
import json
import sys
sys.path.append('/home/can/dictionary-circuits/')
from loading_utils import DictionaryCfg, submodule_name_to_type
from collections import defaultdict
from tqdm import tqdm

# with torch.autograd.profiler.profile(use_cuda=True) as prof:
# Experiment parameters
model_name = "pythia-70m-deduped"
activations_dir = "/home/can/feature_clustering/activations/"
loss_threshold = 0.03
skip = 512 # Trying to ramp this up to choose from diverse documents
n_pos = 10 # Saving feature activations for the final n_pos positions of each context

num_tokens = int(1e4)
batch_size = 1
save_every_n_batches = 100
batch_idxs = torch.arange(save_every_n_batches, 1e4, save_every_n_batches, dtype=torch.int)
batch_idxs -= 1


# Submodule and dictionary parameters
submodules_generic = ['model.gpt_neox.layers.{}.mlp.dense_4h_to_h']
dict_cfg = DictionaryCfg(
    dictionary_size=512 * 64,
    dictionary_dir="/share/projects/dictionary_circuits/autoencoders/pythia-70m-deduped"
)
submod_type_names = "-".join([submodule_name_to_type(s) for s in submodules_generic])
param_summary = f"{model_name}_tloss{loss_threshold}_ntok{num_tokens}_skip{skip}_npos{n_pos}_{submod_type_names}"


#%%
results = defaultdict()
for batch_idx in tqdm(batch_idxs, desc="Result batch read", total=len(batch_idxs)):
    # Load feature activations and gradients on 1k contexts
    act_grad_filename = f"act-n-grad-{batch_idx}_{param_summary}.json"
    act_per_context = json.load(open(os.path.join(activations_dir, act_grad_filename), "r"))
    for k in act_per_context:
        results[k] = act_per_context[k]

json.dump(results, open(os.path.join(activations_dir, f"act-n-grad-cat_{param_summary}.json"), "w"))
# %%
