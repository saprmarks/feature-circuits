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
from loading_utils import load_submodule
from cluster_utils import ClusterConfig, get_tokenized_context_y

# Set enviroment specific constants
device = "cuda:0"
results_dir = "/home/can/feature_clustering/clustering_pythia-70m-deduped_tloss0.03_ntok1024_skip512_npos16_mlp"
tokenized_dataset_dir = "/home/can/data/pile_test_tokenized_600k/"

# Load config, data, model, dictionaries
ccfg = ClusterConfig(**json.load(open(os.path.join(results_dir, "config.json"), "r")))
final_token_idxs = torch.load(os.path.join(results_dir, "final_token_idxs.pt"))
dataset = datasets.load_from_disk(tokenized_dataset_dir)
model = LanguageModel("EleutherAI/"+ccfg.model_name, device_map=device)
submodule = load_submodule(model, "model.gpt_neox.layers.0.mlp.dense_4h_to_h")
n_batches = ccfg.num_samples // ccfg.batch_size

# Data loader
def data_loader(final_token_idxs, batch_size):
    for i in range(n_batches):
        contexts, ys = t.zeros((batch_size, ccfg.n_pos)), t.zeros((batch_size))
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
loader = data_loader(final_token_idxs, ccfg.batch_size)

# Metric
def metric_fn(logits, target_token_id): # logits shape: (batch_size, seq_len, vocab_size)
    m = torch.log_softmax(logits[:, -1, :], dim=-1) # batch_size, vocab_size
    m = m[t.arange(m.shape[0]), target_token_id] # batch_size
    return m.sum()

# Cache feature activations and gradients
for batch_idx in range(n_batches):
    contexts, ys = next(loader) # for batch_size=1
    contexts, ys = contexts.to(device), ys.to(device)

    # model.zero_grad()
    with model.invoke(contexts, fwd_args={'inference': False}) as invoker:
        
        x = submodule.output
        activations = x.detach().save()
        gradients = x.grad.detach().save() # [batch_size, seq_len, vocab_size]
        logits = model.embed_out.output # [batch_size, seq_len, vocab_size]

        met = metric_fn(logits=logits, target_token_id=ys)
        met.backward()

    print(f'gradients : {gradients[0].value.abs().sum()}')


    test_idx = 0
    with model.invoke(contexts[test_idx].unsqueeze(dim=0), fwd_args={'inference': False}) as invoker:
        
        x = submodule.output
        act_check = x.detach().save()
        grad_check = x.grad.detach().save() #Overwrites previous buffer automatically

        logits = model.embed_out.output
        m = metric_fn(logits=logits, target_token_id=ys[test_idx])
        m.backward()

    print(f'gradient_check : {grad_check[0].value.abs().sum()}')

    print(f'diff: {(gradients.value[0] - grad_check[0].value).sum()}\n\n')