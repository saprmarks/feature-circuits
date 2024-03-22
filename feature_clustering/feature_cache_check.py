#%%
# Imports and constants
import os
import torch as t
from nnsight import LanguageModel
from tqdm import trange
import json
import datasets
from collections import defaultdict
import gc
from transformers import AutoTokenizer
import einops
import torch.nn.functional as F

import sys
sys.path.append('/home/can/dictionary-circuits/')
from loading_utils import load_submodules_and_dictionaries_from_generic, DictionaryCfg, submodule_name_to_type
from cluster_utils import ClusterConfig, get_tokenized_context_y

# Set enviroment specific constants
device = "cuda:0"
results_dir = "/home/can/feature_clustering/clustering_pythia-70m-deduped_tloss0.1_nsamples1024_npos32_filtered-induction_mlp-attn-resid"
dictionary_dir="/share/projects/dictionary_circuits/autoencoders/pythia-70m-deduped"
tokenized_dataset_dir = "/home/can/data/pile_test_tokenized_600k/"
model_cache_dir = "/home/can/feature_clustering/model_cache"

# Load config, data, model, dictionaries
ccfg = ClusterConfig(**json.load(open(os.path.join(results_dir, "config.json"), "r")))
final_token_idxs = t.load(os.path.join(results_dir, "final_token_idxs.pt"))
dataset = datasets.load_from_disk(tokenized_dataset_dir)
model = LanguageModel("EleutherAI/"+ccfg.model_name, device_map=device)
dict_cfg = DictionaryCfg(dictionary_size=ccfg.dictionary_size, dictionary_dir=dictionary_dir)
submodule_names, submodules, dictionaries = load_submodules_and_dictionaries_from_generic(model, ccfg.submodules_generic, dict_cfg)


tokenizer = AutoTokenizer.from_pretrained(
        f"EleutherAI/{ccfg.model_name}",
        revision=f"step{143000}",
        cache_dir=os.path.join(model_cache_dir, ccfg.model_name, f"step{143000}"),
    )

#%%

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
loader = data_loader(final_token_idxs, 5)



#%%
arr = t.arange(100)
batch_size = 10
n_batches = len(arr) // batch_size # Not achieving the exact number of samples if num_samples is not divisible by batch_size


def data_test(arr, batch_size):
    for i in range(n_batches):
        contexts = t.zeros((batch_size))
        for j in range(batch_size):
            sample_idx = i * batch_size + j
            contexts[j] = arr[sample_idx] 
        yield contexts.int()
ldr = data_test(arr, batch_size)






#%%
tok_idxs = t.arange(0, 32)
batch_size = 32
batch_ctx = t.zeros((len(tok_idxs), ccfg.n_pos), dtype=t.int)
batch_y = t.zeros((len(tok_idxs), 1), dtype=t.int)

for i, tok_idx in enumerate(tok_idxs):
    context, y, _ = get_tokenized_context_y(
        ccfg,
        dataset,
        int(final_token_idxs[tok_idx, 0]),
        int(final_token_idxs[tok_idx, 1])
    )
    batch_ctx[i] = t.tensor(context)
    batch_y[i] = t.tensor(y)
    print(f"context of final tok {tok_idx}: {tokenizer.decode(context)}\n\n")


# contexts, ys = t.tensor(context).unsqueeze(0).to(device), t.tensor(y).unsqueeze(0).to(device)
activations = t.zeros((ccfg.n_submodules, batch_size, ccfg.n_pos, ccfg.dictionary_size), dtype=t.float32, device=device)
gradients = t.zeros((ccfg.n_submodules, batch_size, ccfg.n_pos, ccfg.dictionary_size), dtype=t.float32, device=device)

# Metric
def metric_fn(logits, target_token_id): # logits shape: (batch_size, seq_len, vocab_size)
    m = t.log_softmax(logits[:, -1, :], dim=-1) # batch_size, vocab_size
    m = m[t.arange(m.shape[0]), target_token_id] # batch_size
    return m.sum()

with model.invoke(batch_ctx, fwd_args={'inference': False}) as invoker:
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
    metric_fn(logits=logits, target_token_id=batch_y).backward()

activations = einops.rearrange(activations, 'n_submodules batch_size n_pos dictionary_size -> batch_size n_pos (n_submodules dictionary_size)')
gradients = einops.rearrange(gradients, 'n_submodules batch_size n_pos dictionary_size -> batch_size n_pos (n_submodules dictionary_size)')

print(activations[0].nonzero().shape)

# %%
pattern1 = activations[0][-1]
pattern2 = activations[1][-1]
cos_sim12 = F.cosine_similarity(pattern1, pattern2, dim=-1)
print(f'cosine similarity between feature pattern vector: {cos_sim12.mean()}')
n1 = pattern1.nonzero()
n2 = pattern2.nonzero()
print(f'number of nonzero elements in both tensors: {n1.shape[0]}, {n2.shape[0]}')
print(f'tensors are equal: {t.all(pattern1 == pattern2)}')
# %%


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
loader = data_loader(final_token_idxs, 20)
# %%
