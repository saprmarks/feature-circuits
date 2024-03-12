"""
This script computes a similarity matrix between samples.
"""

import os
from collections import defaultdict
import pickle
import h5py

import numpy as np
from tqdm.auto import tqdm

import torch as t
import torch.nn.functional as F

import datasets
from transformers import AutoTokenizer, GPTNeoXForCausalLM

# DEFINE SOME PARAMETERS
MODEL_NAME = "pythia-70m-deduped"
STEP = 143_000
CACHE_DIR = f"/om/user/ericjm/pythia-models/{MODEL_NAME}/step{STEP}"
SKIP = 5
N_SAMPLES = 100_000
CHUNK_SIZE = 2_000 # frequency of saving the gradients to hdf5
# BATCH_SIZE = 50
PROJECTION_DIM = 30_000
DENSITY_FACTOR = 16
SAVE_DIR = "/om/user/ericjm/results/dictionary-circuits/dense_clustering/exp008"
device = t.device('cuda:0') if t.cuda.is_available() else 'cpu'
t.set_default_dtype(t.float32)

assert N_SAMPLES % CHUNK_SIZE == 0, "N_SAMPLES must be a multiple of CHUNK_SIZE"

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

class SparseProjectionOperator:
    """
    Note: I think the sparsity is off by a factor of two here.
    """
    def __init__(self, original_dim, projection_dim, sparsity, seed=0, device='cpu'):
        t.manual_seed(seed)
        t.cuda.manual_seed(seed) # if 'cuda' in device else None
        self.device = t.device(device)
        self.original_dim = original_dim
        self.lambda_ = original_dim * (1 - sparsity)
        num_entries = t.poisson(self.lambda_ * t.ones(projection_dim, device=device)).int()
        max_entries = num_entries.max()
        self.positives = t.randint(0, original_dim, (projection_dim, max_entries), device=device)
        self.negatives = t.randint(0, original_dim, (projection_dim, max_entries), device=device)
        masks = t.arange(max_entries, device=device).expand(projection_dim, max_entries) < num_entries.unsqueeze(-1)
        self.positives = self.positives * masks
        self.negatives = self.negatives * masks

    def __call__(self, x):
        # assert x.device == self.device, "device mismatch between projection and input"
        assert x.shape[-1] == self.original_dim, "input dimension mismatch"
        y = x[self.positives].sum(-1) - x[self.negatives].sum(-1)
        return y

############################
# Select samples to cluster
############################
with open("/om2/user/ericjm/dictionary-circuits/dense_clustering/experiments/exp008/zero_and_induction_idxs-pythia-70m-deduped.pkl", "rb") as f:
    non_induction_zeros, zero_idxs, induction_idxs = pickle.load(f)
idxs = non_induction_zeros[::SKIP][:N_SAMPLES]

if len(idxs) != N_SAMPLES:
    print(f"ERROR: only {len(idxs)} samples available")
    exit()


def get_flattened_gradient(model, param_subset):
    grads = []
    for name, p in model.named_parameters():
        if name in param_subset:
            grads.append(p.grad)
    return t.cat([g.flatten() for g in grads])

param_names = [n for n, _ in model.named_parameters()]
highsignal_names = [name for name in param_names if 
                        ('layernorm' not in name) and 
                        ('embed' not in name)]

len_g = sum(model.state_dict()[name].numel() for name in highsignal_names)
print(f"len_g = {len_g}")

sparse_projector = SparseProjectionOperator(len_g, PROJECTION_DIM, 1 - (DENSITY_FACTOR / PROJECTION_DIM), device=device)

############################
# Compute similarity matrix
############################
Gs = np.zeros((CHUNK_SIZE, PROJECTION_DIM), dtype=np.float32)
# blocks = [idxs[i:min(len(idxs), i+BLOCK_SIZE)] for i in range(0, len(idxs), BLOCK_SIZE)]
model.eval()
with h5py.File(os.path.join(SAVE_DIR, f"gradients.h5"), "a") as f:
    h5dset = f.create_dataset("gradients", (N_SAMPLES, PROJECTION_DIM), 
        chunks=(CHUNK_SIZE, PROJECTION_DIM), dtype=np.float32)

    for i, idx in enumerate(tqdm(idxs)):
        model.zero_grad()
        document, l = get_context(idx)
        prompt = document['text']
        tokens = tokenizer(prompt, return_tensors='pt', max_length=1024, truncation=True).to(device)
        logits = model(**tokens).logits
        targets = tokens.input_ids
        ls = t.nn.functional.cross_entropy(logits[0, :-1, :], targets[0, 1:], reduction='none')
        ls_l = ls[l-1]
        ls_l.backward()
        g = get_flattened_gradient(model, highsignal_names)
        g_projected = sparse_projector(g)
        Gs[i % CHUNK_SIZE] = g_projected.detach().cpu().numpy()
        if i % CHUNK_SIZE == CHUNK_SIZE - 1:
            h5dset[i-CHUNK_SIZE+1:i+1] = Gs
            Gs = np.zeros((CHUNK_SIZE, PROJECTION_DIM), dtype=np.float32)

# save the idxs
with open(os.path.join(SAVE_DIR, "idxs.pkl"), "wb") as f:
    pickle.dump(idxs, f)
