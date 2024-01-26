# %%
import json
import numpy as np
import sys
import os
from tqdm import tqdm
sys.path.append("/home/can/")
from feature_clustering.streamlit.cluster_exploration import *

results_dir = "/home/can/feature_clustering/activations/"
context_dir = "/home/can/feature_clustering/contexts"
def load_act_n_grad_results(filename = "act-n-grad_pythia-70m-deduped_loss-thresh0.005_skip50_ntok10000_nonzero_pos-reduction-final_mlp.json"):
    path = os.path.join(results_dir, filename)
    act_per_context = json.loads(open(path).read())
    y_global_idx = np.array(list(act_per_context.keys()), dtype=int)
    num_y = len(act_per_context)
    return act_per_context, y_global_idx, num_y

act_per_context, y_global_idx, num_y = load_act_n_grad_results()

# For each global_idx in act_per_context, save true tokens and contexts to dictionary
y_contexts = {}

for global_idx in tqdm(act_per_context, total=num_y, desc="Saving contexts"):
    global_idx = int(global_idx)
    doc_idx = loss_idx_to_dataset_idx(global_idx)[0]
    document, token_idx = get_context(global_idx)
    # get true token y and the preceding context of at most 100 tokens
    y = document["split_by_token"][token_idx]
    context = document["split_by_token"][max(0, token_idx-100):token_idx]
    context = "".join(context)
    y_contexts[global_idx] = dict(y=y, context=context, document_idx=doc_idx)

# Save to file
context_filename = "contexts_pythia-70m-deduped_loss-thresh0.005_skip50_ntok10000_nonzero_pos-reduction-final_mlp.json"
context_path = os.path.join(context_dir, context_filename)
with open(context_path, "w") as f:
    json.dump(y_contexts, f)

# %%
y_contexts[5896934]
# %%
