"""
In this script we'll create the `non_induction_idxs.pkl` file equivalent for 
pythia-70m-deduped (rather than pythia-70m-deduped-v0) like we already have.
"""

from collections import defaultdict
import pickle

import numpy as np
from tqdm.auto import tqdm

import datasets


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


# load up losses for pythia-70m-deduped
losses = np.load("/om/user/ericjm/results/trajectory-dictionaries/pythia-70m-deduped-20k/pythia-70m-deduped-step143000.npy")

zero_idxs = losses < 0.3
zero_idxs, = zero_idxs.nonzero()
print(len(zero_idxs))
print_context(zero_idxs[0])

# induction_idxs = np.zeros((curves.shape[0],), dtype='bool')
induction_idxs = []
i = 0
for document_idx in tqdm(range(20_000)):
    document = dataset[document_idx]
    document_trigrams = defaultdict(int)
    tokens = document['input_ids'][0]
    if len(tokens) > 1:
        i += 1
        for j in range(2, len(tokens)):
            trigram = tuple(tokens[j-2:j+1])
            if trigram in document_trigrams:
                # induction_idxs[i] = 1
                induction_idxs.append(i)
            document_trigrams[trigram] += 1
            i += 1

non_induction_zeros = set(zero_idxs).difference(set(induction_idxs))
non_induction_zeros = sorted(list(non_induction_zeros))
print(f"found {len(non_induction_zeros)} non-induction zeros")

# save idxs
with open("zero_and_induction_idxs-pythia-70m-deduped.pkl", "wb") as f:
    pickle.dump((non_induction_zeros, zero_idxs, induction_idxs), f)

