import json
import numpy as np
import sys
import os
sys.path.append("/home/can/")
from feature_clustering.streamlit.cluster_exploration import *

results_dir = "/home/can/feature_clustering/results"
def load_act_n_grad_results(filename = "act-n-grad_pythia-70m-deduped_loss-thresh0.005_skip50_ntok10000_nonzero_pos-reduction-final_mlp.json"):
    path = os.path.join(results_dir, filename)
    act_per_context = json.loads(open(path).read())
    y_global_idx = np.array(list(act_per_context.keys()), dtype=int)
    num_y = len(act_per_context)
    return act_per_context, y_global_idx, num_y

act_per_context, y_global_idx, num_y = load_act_n_grad_results()

# For each global_idx in act_per_context, save true tokens and contexts to dictionary
y_contexts = {}

for global_idx in act_per_context:
    global_idx = int(global_idx)
    sample, token_idx = get_context(global_idx)
    # get true token y and the preceding context of at most 100 tokens
    y = sample["split_by_token"][token_idx]
    context = sample["split_by_token"][max(0, token_idx-100):token_idx]
    context = "".join(context)
    y_contexts[global_idx] = dict(y=y, context=context)

# Save to file
context_filename = "contexts_pythia-70m-deduped_loss-thresh0.005_skip50_ntok10000_nonzero_pos-reduction-final_mlp.json"
context_path = os.path.join(results_dir, context_filename)
with open(context_path, "w") as f:
    json.dump(y_contexts, f)

#%%
10e4 * (1.5e3 * 12 + 5e2 * 16 * 10) / 1024**3 

# %%
