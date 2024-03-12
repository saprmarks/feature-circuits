"""
This script computes a similarity matrix between samples.
"""

import os
from collections import defaultdict
import pickle

import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
import sklearn.cluster

import datasets
from transformers import AutoTokenizer, GPTNeoXForCausalLM
# import transformer_lens


# DEFINE SOME PARAMETERS
MODEL_NAME = "pythia-70m-v0"
STEP = 143000
CACHE_DIR = f"/om/user/ericjm/pythia-models/{MODEL_NAME}/step{STEP}"
SKIP = 50
N_SAMPLES = 10000
BLOCK_SIZE = 2000
BATCH_SIZE = 100
SAVE_DIR = "/om/user/ericjm/results/dictionary-circuits/dense_clustering/exp002"
device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

assert N_SAMPLES % BLOCK_SIZE == 0, "N_SAMPLES must be divisible by BLOCK_SIZE"
assert BLOCK_SIZE % BATCH_SIZE == 0, "BLOCK_SIZE must be divisible by BATCH_SIZE"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


######################
# Load up the dataset
######################
pile_canonical = "/om/user/ericjm/the_pile/the_pile_test_canonical_200k"
dataset = datasets.load_from_disk(pile_canonical)

starting_indexes = np.array([0] + list(np.cumsum(dataset["preds_len"])))

def loss_idx_to_dataset_idx(idx):
    """given an idx, return a document index and pred-in-sample
    index in range(0, 1023). Note token-in-sample idx is
    exactly pred-in-sample + 1. So the pred_in_sample_index is the index
    into the sequence above will the model will genenerate a prediction for the
    token at the pred_in_sample_index + 1."""
    sample_index = np.searchsorted(starting_indexes, idx, side="right") - 1
    pred_in_sample_index = idx - starting_indexes[sample_index]
    return int(sample_index), int(pred_in_sample_index)

def get_context(idx):
    """given idx, return dataset document and the index of the token 
    corresponding to the given idx within that document, in range(1, 1024)."""
    sample_index, pred_index = loss_idx_to_dataset_idx(idx)
    return dataset[sample_index], pred_index+1

def print_context(idx, context_length=-1):
    """
    given idx, print the context preceding the corresponding
    token as well as the token itself, and highlight the token.
    """
    sample, token_idx = get_context(idx)
    prompt = sample["split_by_token"][:token_idx]
    if context_length > 0:
        prompt = prompt[-context_length:]
    prompt = "".join(prompt)
    token = sample["split_by_token"][token_idx]
    print(prompt + "\033[41m" + token + "\033[0m")

##############################
# Load up model and tokenizer
##############################
model = GPTNeoXForCausalLM.from_pretrained(
        f"EleutherAI/{MODEL_NAME}",
        revision=f"step{STEP}",
        cache_dir=CACHE_DIR,
    ).to(device)

tokenizer = AutoTokenizer.from_pretrained(
    f"EleutherAI/{MODEL_NAME}",
    revision=f"step{STEP}",
    cache_dir=CACHE_DIR,
)
tokenizer.pad_token = tokenizer.eos_token

############################
# Select samples to cluster
############################
with open("/om2/user/ericjm/the-everything-machine/tmp/zero_and_induction_idxs.pkl", "rb") as f:
    non_induction_zeros, zero_idxs, induction_idxs = pickle.load(f)
idxs = non_induction_zeros[::SKIP][:N_SAMPLES]

if len(idxs) != N_SAMPLES:
    print(f"WARNING: only {len(idxs)} samples available")

############################
# Compute similarity matrix
############################
d = model.config.hidden_size * model.config.num_hidden_layers # cluster by residual streams
blocks = [idxs[i:min(len(idxs), i+BLOCK_SIZE)] for i in range(0, len(idxs), BLOCK_SIZE)]
C = torch.zeros((len(idxs), len(idxs)), device=device)
model.eval()
with torch.inference_mode():
    iouter = 0
    for iblock in tqdm(blocks):
        Fi = torch.zeros((len(iblock), d), device=device)
        # break the block into batches
        for ibatch_start in range(0, len(iblock), BATCH_SIZE):
            batch_is = list(range(ibatch_start, min(len(iblock), ibatch_start+BATCH_SIZE)))
            batch_idxs = [iblock[i] for i in batch_is]
            batch_prompts = []
            batch_ls = []
            for idx in batch_idxs:
                document, l = get_context(idx)
                prompt = document['text']
                batch_prompts.append(prompt)
                batch_ls.append(l)
            tokens = tokenizer(batch_prompts, return_tensors='pt', padding=True, max_length=1024, truncation=True).to(device)

            # hook the model
            module_outputs = {}
            module_inputs = {}
            def named_hook_inputs(name):
                def hook_fn(module, input, output):
                    module_inputs[name] = input[0]
                return hook_fn
            def named_hook_outputs(name):
                def hook_fn(module, input, output):
                    module_outputs[name] = output
                return hook_fn
            for name, module in model.named_modules():
                if name: # empty string is the top-level module
                    module.register_forward_hook(named_hook_inputs(name))
                    module.register_forward_hook(named_hook_outputs(name))
            
            # forward the model
            _ = model(**tokens, output_hidden_states=True)

            seq_poss = torch.tensor([l-1 for l in batch_ls], dtype=torch.long, device=device)
            batch_dim_range = torch.arange(len(batch_is), device=device)
            # get the features
            features = torch.cat([module_inputs[f'gpt_neox.layers.{li}.input_layernorm'][batch_dim_range, seq_poss] for li in range(model.config.num_hidden_layers)], dim=1)
            # concatenate the features
            Fi[batch_is] = features
        Fi = F.normalize(Fi, p=2, dim=1)
        j_index = blocks.index(iblock)
        jouter = sum(len(block) for block in blocks[:j_index])
        for jblock in tqdm(blocks[j_index:], leave=False):
            Fj = torch.zeros((len(jblock), d), device=device)

            for ibatch_start in range(0, len(jblock), BATCH_SIZE):
                batch_is = list(range(ibatch_start, min(len(jblock), ibatch_start+BATCH_SIZE)))
                batch_idxs = [jblock[i] for i in batch_is]
                batch_prompts = []
                batch_ls = []
                for idx in batch_idxs:
                    document, l = get_context(idx)
                    prompt = document['text']
                    batch_prompts.append(prompt)
                    batch_ls.append(l)
                tokens = tokenizer(batch_prompts, return_tensors='pt', padding=True, max_length=1024, truncation=True).to(device)

                # hook the model
                module_outputs = {}
                module_inputs = {}
                def named_hook_inputs(name):
                    def hook_fn(module, input, output):
                        module_inputs[name] = input[0]
                    return hook_fn
                def named_hook_outputs(name):
                    def hook_fn(module, input, output):
                        module_outputs[name] = output
                    return hook_fn
                for name, module in model.named_modules():
                    if name: # empty string is the top-level module
                        module.register_forward_hook(named_hook_inputs(name))
                        module.register_forward_hook(named_hook_outputs(name))

                # forward the model
                _ = model(**tokens, output_hidden_states=True)

                seq_poss = torch.tensor([l-1 for l in batch_ls], dtype=torch.long, device=device)
                batch_dim_range = torch.arange(len(batch_is), device=device)
                # get the features
                features = torch.cat([module_inputs[f'gpt_neox.layers.{li}.input_layernorm'][batch_dim_range, seq_poss] for li in range(model.config.num_hidden_layers)], dim=1)
                Fj[batch_is] = features
            Fj = F.normalize(Fj, p=2, dim=1)
            Cij = torch.matmul(Fi, Fj.T)
            C[iouter:iouter+len(iblock), jouter:jouter+len(jblock)] = Cij
            C[jouter:jouter+len(jblock), iouter:iouter+len(iblock)] = Cij.T
            jouter += len(jblock)
        iouter += len(iblock)

torch.save((idxs, C.cpu()), os.path.join(SAVE_DIR, "similarity.pt"))


# NOTES ON NAMING CONVENTION FOR MODULES
# 'gpt_neox.embed_in' has input a tensor of shape (batch_size, sequence_length) of integers
# 'gpt_neox.embed_in' has output a tensor of shape (batch_size, sequence_length, hidden_size)
# 'gpt_neox.layers.0.input_layernorm receives as input the output of 'gpt_neox.embed_in'

# the input of 'gpt_neox.layers.0.post_attention_layernorm' must be the residual stream after the attention block
# the output of 'gpt_neox.layers.0.post_attention_layernorm' is the same as the input of 'gpt_neox.layers.0.mlp'

# alright looking closely at the code and the v0 model, it seems that it uses `use_parallel_residual` which means that the layer as a whole performs:
# x = x + attn(ln1(x)) + mlp(ln2(x)) rather than:
# x = x + attn(ln1(x))
# x = x + mlp(ln2(x))
# we find that the input of 'gpt_neox.layers.1.input_layernorm' is equal to the output of 'gpt_neox.embed_in' + the output of 'gpt_neox.layers.0.attention[0]' + the output of 'gpt_neox.layers.0.mlp'

# module_outputs['gpt_neox.layers.{i}.mlp.act] # MLP post-activations
# module_outputs[f'gpt_neox.layers.{i}.mlp'] # what MLP writes to residual stream
# module_outputs[f'gpt_neox.layers.{i}.attention'][0] # what attention writes to residual stream
# module_inputs[f'gpt_neox.layers.{i}.input_layernorm'] # residual stream right before layer i
