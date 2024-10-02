import json
import random
import torch as t
import torch.nn.functional as F
from nnsight import LanguageModel


def load_examples(
    dataset: str,
    n_examples: int,
    model: LanguageModel,
    use_min_length_only: bool = False,
    max_length: int = None,
    seed: int = 12
):
    with open(dataset, "r") as f:
        dataset_items = f.readlines()
    random.Random(seed).shuffle(dataset_items)

    examples = []
    if use_min_length_only:
        min_length = float("inf")
    for line in dataset_items:
        data = json.loads(line)
        clean_prefix = model.tokenizer(data["clean_prefix"]).input_ids
        patch_prefix = model.tokenizer(data["patch_prefix"]).input_ids
        clean_answer = model.tokenizer(data["clean_answer"]).input_ids
        patch_answer = model.tokenizer(data["patch_answer"]).input_ids
        clean_full = model.tokenizer(data["clean_prefix"] + data["clean_answer"]).input_ids
        patch_full = model.tokenizer(data["patch_prefix"] + data["patch_answer"]).input_ids

        # strip BOS token from response if necessary
        if clean_answer[0] == model.tokenizer.bos_token_id:
            clean_answer = clean_answer[1:]
        if patch_answer[0] == model.tokenizer.bos_token_id:
            patch_answer = patch_answer[1:]
        
        # check that answer is one token
        if len(clean_answer) != 1 or len(patch_answer) != 1:
            continue

        # check that prefixes are the same length
        if len(clean_prefix) != len(patch_prefix):
            continue

        # check for tokenization mismatches
        if clean_prefix + clean_answer != clean_full:
            continue
        if patch_prefix + patch_answer != patch_full:
            continue

        if max_length is not None and len(clean_prefix) > max_length:
            continue

        if use_min_length_only:
            # restart collection if we've found a new shortest example
            if (l := len(clean_prefix)) < min_length:
                examples = [] # restart collection
                min_length = l
            # skip if too long
            elif l > min_length:
                continue
        
        examples.append(data)

        if len(examples) >= n_examples:
            return examples

def load_examples_nopair(*args, **kwargs):
    pass

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