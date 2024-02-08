import os
import numpy as np
import torch
import torch.nn.functional as F
import datasets
from dataclasses import dataclass
import sys
sys.path.append('../')

@dataclass
class ClusterConfig():
    def __init__(self, 
                 model_name, 
                 loss_threshold, 
                 num_samples,
                 skip, 
                 n_pos, 
                 submodules_generic,
                 dictionary_size,
                 results_dir=None,
                 ):
        self.model_name = model_name
        self.loss_threshold = loss_threshold
        self.num_samples = num_samples
        self.skip = skip # Trying to ramp this up to choose from diverse documents
        self.n_pos = n_pos
        self.submodules_generic = submodules_generic
        self.dictionary_size = dictionary_size
        self.results_dir = results_dir

def loss_idx_to_dataset_idx(idx, starting_indexes):
    """given an idx in range(0, 10658635), return
    a document index in range(0, 600000) and pred-in-document
    index in range(0, 1023). Note token-in-document idx is
    exactly pred-in-document + 1"""
    document_index = np.searchsorted(starting_indexes, idx, side="right") - 1
    final_token_in_context_index = idx - starting_indexes[document_index]
    return int(document_index), int(final_token_in_context_index)

def get_tokenized_context_y(ccfg, dataset, doc_idx, final_token_in_context_index):
    """The length of idxs determines the batch size. Given idx in range(0, 10658635)
    and predicted token within sample, in range(1, 1024)."""
    y_idx = final_token_in_context_index + 1
    context_start_idx = y_idx - ccfg.n_pos
    context = dataset[doc_idx]['input_ids'][0][context_start_idx:y_idx]
    y = dataset[doc_idx]['input_ids'][0][y_idx]
    return context, y, doc_idx

def get_string_context_y(tokenizer, ccfg, dataset, doc_idx, final_token_in_context_index):
    context, y, doc_idx = get_tokenized_context_y(ccfg, dataset, doc_idx, final_token_in_context_index)
    concatenated_context = "".join(tokenizer.decode(context))
    return concatenated_context, tokenizer.decode(y), doc_idx


class LossesDataset():
    def __init__(self, cluster_config, model_cache_dir, tokenized_dataset_dir, batch_size, left_pad_to_length=1024):
        self.model_name = cluster_config.model_name
        self.model_cache_dir = model_cache_dir
        self.tokenized_dataset_dir = tokenized_dataset_dir
        self.loss_threshold = cluster_config.loss_threshold
        self.c = 1 / np.log(2)
        self.num_tokens = cluster_config.num_tokens
        self.batch_size = batch_size
        self.skip = cluster_config.skip
        self.left_pad_to_length = left_pad_to_length

        self.n_batches = self.num_tokens // self.batch_size
        self.dataset = datasets.load_from_disk(tokenized_dataset_dir)
        self.starting_indexes = np.array([0] + list(np.cumsum(self.dataset["preds_len"])))
        self.token_loss_idxs = self._load_token_loss_idxs()

    def _get_tokenized_contexts_y(self, idxs):
        """The length of idxs determines the batch size. Given idx in range(0, 10658635), return dataset sample padded to self.left_pad_to_length tokens
        and predicted token within sample, in range(1, 1024)."""
        contexts = torch.zeros((len(idxs), self.left_pad_to_length), dtype=torch.int)
        ys = torch.zeros((len(idxs)), dtype=torch.int)
        for i, idx in enumerate(idxs):
            sample_index, pred_index = self._loss_idx_to_dataset_idx(idx)
            context = torch.tensor(self.dataset[sample_index]['input_ids'][0][:pred_index+1], device="cpu") # This was not sliced with :pred_index+1 in quanta-discovery. Bug?
            padded_context = F.pad(context, (self.left_pad_to_length - len(context), 0), value=0) # Left padding with value 0, padding token of GPTNeoXTokenizer
            contexts[i] = padded_context
            ys[i] = self.dataset[sample_index]['input_ids'][0][pred_index+1]
        return contexts, ys, idxs
    
    def generator(self):
        for i in range(self.n_batches):
            idxs = self.token_loss_idxs[i*self.batch_size:(i+1)*self.batch_size]
            contexts, ys, idxs = self._get_tokenized_contexts_y(idxs)
            yield contexts, ys, idxs