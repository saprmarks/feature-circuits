import os
import numpy as np
import torch
import torch.nn.functional as F
import datasets


class LossesDataset():
    def __init__(self, model, model_name, model_cache_dir, tokenized_dataset_dir, loss_threshold, num_tokens, skip, batch_size, left_pad_to_length=1024):
        self.model = model
        self.model_name = model_name
        self.model_cache_dir = model_cache_dir
        self.tokenized_dataset_dir = tokenized_dataset_dir
        self.loss_threshold = loss_threshold
        self.c = 1 / np.log(2)
        self.num_tokens = num_tokens
        self.batch_size = batch_size
        self.skip = skip
        self.left_pad_to_length = left_pad_to_length

        self.n_batches = self.num_tokens // self.batch_size
        self.dataset = datasets.load_from_disk(tokenized_dataset_dir)
        self.starting_indexes = np.array([0] + list(np.cumsum(self.dataset["preds_len"])))
        self.token_loss_idxs = self._load_token_loss_idxs()

    def _load_token_loss_idxs(self):
        # Load losses data
        particular_model_cache_dir = os.path.join(self.model_cache_dir, self.model_name)
        losses_cached = [f for f in os.listdir(particular_model_cache_dir) if f.endswith("losses.pt")]
        max_i = max(list(range(len(losses_cached))), key=lambda i: int(losses_cached[i].split("_")[0]))
        docs, tokens = int(losses_cached[max_i].split("_")[0]), int(losses_cached[max_i].split("_")[2])
        losses = torch.load(os.path.join(particular_model_cache_dir, f"{docs}_docs_{tokens}_tokens_losses.pt"))
        c = 1 / np.log(2) # for nats to bits conversion

        token_loss_idxs = (losses < (self.loss_threshold / c)).nonzero().flatten()
        token_loss_idxs = token_loss_idxs[::self.skip]
        token_loss_idxs = token_loss_idxs[:self.num_tokens].tolist()
        assert len(token_loss_idxs) == self.num_tokens, "not enough tokens to analyze"
        return token_loss_idxs

    def _loss_idx_to_dataset_idx(self, idx):
        """given an idx in range(0, 10658635), return
        a sample index in range(0, 20000) and pred-in-sample
        index in range(0, 1023). Note token-in-sample idx is
        exactly pred-in-sample + 1"""
        sample_index = np.searchsorted(self.starting_indexes, idx, side="right") - 1
        pred_in_sample_index = idx - self.starting_indexes[sample_index]
        return int(sample_index), int(pred_in_sample_index)

    def _get_tokenized_contexts_y(self, idxs):
        """The length of idxs determines the batch size. Given idx in range(0, 10658635), return dataset sample padded to self.left_pad_to_length tokens
        and predicted token within sample, in range(1, 1024)."""
        contexts = torch.zeros((len(idxs), self.left_pad_to_length), dtype=torch.int)
        ys = torch.zeros((len(idxs)), dtype=torch.int)
        for i, idx in enumerate(idxs):
            sample_index, pred_index = self._loss_idx_to_dataset_idx(idx)
            context = torch.tensor(self.dataset[sample_index]['input_ids'][0][:pred_index+1], device="cpu") # This was not sliced with :pred_index+1 in quanta-discovery. Bug?
            padded_context = F.pad(context, (self.left_pad_to_length - len(context), 0), value=self.model.tokenizer.pad_token_id) # Left padding
            contexts[i] = padded_context
            ys[i] = self.dataset[sample_index]['input_ids'][0][pred_index+1]
        return contexts, ys
    
    def generator(self):
        for i in range(self.n_batches):
            idxs = self.token_loss_idxs[i*self.batch_size:(i+1)*self.batch_size]
            contexts, ys = self._get_tokenized_contexts_y(idxs)
            yield contexts, ys