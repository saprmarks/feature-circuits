import os
import re
import json
import random
import torch as t
import torch.nn.functional as F
from dictionary_learning.dictionary import AutoEncoder
from dataclasses import dataclass

@dataclass
class DictionaryCfg():
    def __init__(
        self,
        dictionary_dir,
        dictionary_size
        ) -> None:
        self.dir = dictionary_dir
        self.size = dictionary_size


def load_examples(dataset, num_examples, model, seed=12, pad_to_length=None, length=None):
    examples = []
    dataset_items = open(dataset).readlines()
    random.seed(seed)
    random.shuffle(dataset_items)
    breakpoint()
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
        breakpoint()
        # only keep examples where clean and patch inputs are the same length
        if clean_prefix.shape[1] != patch_prefix.shape[1]:
            continue
        # only keep examples where answers are single tokens
        breakpoint()
        if clean_answer.shape[1] != 1 or patch_answer.shape[1] != 1:
            continue
        breakpoint()
        # if we specify a `length`, filter examples if they don't match
        if length and clean_prefix.shape[1] != length:
            continue
        breakpoint()
        # if we specify `pad_to_length`, left-pad all inputs to a max length
        prefix_length_wo_pad = clean_prefix.shape[1]
        if pad_to_length:
            model.tokenizer.padding_side = 'right'
            pad_length = pad_to_length - prefix_length_wo_pad
            breakpoint()
            if pad_length < 0:  # example too long
                continue
            breakpoint()
            # left padding: reverse, right-pad, reverse
            clean_prefix = t.flip(F.pad(t.flip(clean_prefix, (1,)), (0, pad_length), value=model.tokenizer.pad_token_id), (1,))
            patch_prefix = t.flip(F.pad(t.flip(patch_prefix, (1,)), (0, pad_length), value=model.tokenizer.pad_token_id), (1,))
        breakpoint()
        example_dict = {"clean_prefix": clean_prefix,
                        "patch_prefix": patch_prefix,
                        "clean_answer": clean_answer.item(),
                        "patch_answer": patch_answer.item(),
                        "annotations": get_annotation(dataset, model, data),
                        "prefix_length_wo_pad": prefix_length_wo_pad,}
        examples.append(example_dict)
        breakpoint()
        if len(examples) >= num_examples:
            break

    return examples


def load_examples_nopair(dataset, num_examples, model, length=None):
    examples = []
    if isinstance(dataset, str):        # is a path to a .json file
        dataset = json.load(open(dataset))
    elif isinstance(dataset, dict):     # is an already-loaded dictionary
        pass
    else:
        raise ValueError(f"`dataset` is unrecognized type: {type(dataset)}. Must be path (str) or dict")
    
    max_len = 0     # for padding
    for context_id in dataset:
        context = dataset[context_id]["context"]
        if length is not None and len(context) > length:
            context = context[-length:]
        clean_prefix = model.tokenizer("".join(context), return_tensors="pt",
                        padding=False).input_ids
        max_len = max(max_len, clean_prefix.shape[-1])

    for context_id in dataset:
        answer = dataset[context_id]["answer"]
        context = dataset[context_id]["context"]
        clean_prefix = model.tokenizer("".join(context), return_tensors="pt",
                                    padding=False).input_ids
        clean_answer = model.tokenizer(answer, return_tensors="pt",
                                    padding=False).input_ids
        if clean_answer.shape[1] != 1:
            continue
        prefix_length_wo_pad = clean_prefix.shape[1]
        pad_length = max_len - prefix_length_wo_pad
        # left padding: reverse, right-pad, reverse
        clean_prefix = t.flip(F.pad(t.flip(clean_prefix, (1,)), (0, pad_length), value=model.tokenizer.pad_token_id), (1,))

        example_dict = {"clean_prefix": clean_prefix,
                        "clean_answer": clean_answer.item(),
                        "prefix_length_wo_pad": prefix_length_wo_pad,}
        examples.append(example_dict)
        if len(examples) >= num_examples:
            break

    return examples

def get_annotation(dataset, model, data):
    # First, understand which dataset we're working with
    structure = None
    if "within_rc" in dataset:
        structure = "within_rc"
        template = "the_subj subj_main that the_dist subj_dist"
    elif "rc.json" in dataset or "rc_" in dataset:
        structure = "rc"
        template = "the_subj subj_main that the_dist subj_dist verb_dist"
    elif "simple.json" in dataset or "simple_" in dataset:
        structure = "simple"
        template = "the_subj subj_main"
    elif "nounpp.json" in dataset or "nounpp_" in dataset:
        structure = "nounpp"
        template = "the_subj subj_main prep the_dist subj_dist"

    if structure is None:
        return {}
    
    annotations = {}

    # Iterate through words in the template and input. Get token spans
    curr_token = 0
    for template_word, word in zip(template.split(), data["clean_prefix"].split()):
        if word != "The":
            word = " " + word
        word_tok = model.tokenizer(word, return_tensors="pt", padding=False).input_ids
        num_tokens = word_tok.shape[1]
        span = (curr_token, curr_token + num_tokens-1)
        curr_token += num_tokens
        annotations[template_word] = span
    
    return annotations