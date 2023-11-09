import zstandard as zstd
import json
import os
import io
import random
from tqdm import tqdm
from nnsight import LanguageModel
from dictionary_learning.buffer import ActivationBuffer
from dictionary_learning.dictionary import AutoEncoder
from dictionary_learning.training import trainSAE
from datasets import load_dataset
from einops import rearrange
import torch as t

def search_for_phenomenon(model, submodule, autoencoder, buffer,
                          dataset, return_examples=None, seed=12,
                          num_examples=100):
    def _load_examples(dataset, num_examples):
        examples = []
        dataset_items = open(dataset).readlines()
        random.seed(seed)
        random.shuffle(dataset_items)
        for line in dataset_items:
            data = json.loads(line)
            clean_prefix = model.tokenizer(data["clean_prefix"], return_tensors="pt",
                                           padding=False).input_ids
            patch_prefix = model.tokenizer(data["patch_prefix"], return_tensors="pt",
                                           padding=False).input_ids
            clean_answer = model.tokenizer(data["clean_answer"], return_tensors="pt",
                                           padding=False).input_ids
            patch_answer = model.tokenizer(data["patch_answer"], return_tensors="pt",
                                           padding=False).input_ids
            if clean_prefix.shape[1] != patch_prefix.shape[1]:
                continue
            if clean_answer.shape[1] != 1 or patch_answer.shape[1] != 1:
                continue
            
            example_dict = {"clean_prefix": clean_prefix, "patch_prefix": patch_prefix,
                            "clean_answer": clean_answer, "patch_answer": patch_answer}
            examples.append(example_dict)
            if len(examples) >= num_examples:
                break
        return examples

    examples = _load_examples(dataset, num_examples)
    # TODO: implement indirect effects for features
    print(f"Number of examples: {len(examples)}")
    print(examples[0])