import os
import re
import json
import random
import torch as t
import torch.nn.functional as F
from dictionary_learning.dictionary import AutoEncoder
from dataclasses import dataclass

@dataclass
class DictionaryCfg(): # TODO Move to dictionary_learning repo?
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
        if length and clean_prefix.shape[1] != length:
            continue
        prefix_length_wo_pad = clean_prefix.shape[1]
        if pad_to_length:
            model.tokenizer.padding_side = 'right' # TODO: move this after model initialization
            pad_length = pad_to_length - prefix_length_wo_pad
            if pad_length < 0:  # example too long
                continue
            clean_prefix = F.pad(clean_prefix, (0, pad_length), value=model.tokenizer.pad_token_id)
            patch_prefix = F.pad(patch_prefix, (0, pad_length), value=model.tokenizer.pad_token_id)
        example_dict = {"clean_prefix": clean_prefix,
                        "patch_prefix": patch_prefix,
                        "clean_answer": clean_answer.item(),
                        "patch_answer": patch_answer.item(),
                        "annotations": get_annotation(dataset, model, data),
                        "prefix_length_wo_pad": prefix_length_wo_pad,}
        examples.append(example_dict)
        if len(examples) >= num_examples:
            break
    return examples


def load_examples_nopair(dataset, num_examples, model):
    examples = []
    dataset_json = json.load(open(dataset))
    max_len = 0     # for padding
    for context_id in dataset_json:
        context = dataset_json[context_id]["context"]
        clean_prefix = model.tokenizer("".join(context), return_tensors="pt",
                        padding=False).input_ids
        max_len = max(max_len, clean_prefix.shape[-1])

    for context_id in dataset_json:
        answer = dataset_json[context_id]["answer"]
        context = dataset_json[context_id]["context"]
        clean_prefix = model.tokenizer("".join(context), return_tensors="pt",
                                    padding=False).input_ids
        clean_answer = model.tokenizer(answer, return_tensors="pt",
                                    padding=False).input_ids
        if clean_answer.shape[1] != 1:
            continue
        prefix_length_wo_pad = clean_prefix.shape[1]
        model.tokenizer.padding_side = 'right' # TODO: move this after model initialization
        pad_length = max_len - prefix_length_wo_pad
        clean_prefix = F.pad(clean_prefix, (0, pad_length), value=model.tokenizer.pad_token_id)
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


def load_submodule(model, submodule_str):
    if "." not in submodule_str:
        return getattr(model, submodule_str)
    
    submodules = submodule_str.split(".")
    curr_module = None
    for module in submodules:
        if module == "model":
            continue
        if not curr_module:
            curr_module = getattr(model, module)
            continue
        curr_module = getattr(curr_module, module)
    return curr_module


def submodule_type_to_name(submodule_type):
    if submodule_type == "mlp":
        return "model.gpt_neox.layers.{}.mlp.dense_4h_to_h"
    elif submodule_type == "attn":
        return "model.gpt_neox.layers.{}.attention.dense"
    elif submodule_type.startswith("resid"):
        return "model.gpt_neox.layers.{}"
    raise ValueError("Unrecognized submodule type. Please select from {mlp, attn, resid}")


def submodule_name_to_type_layer(submod_name):
    layer_match = re.search(r"layers\.(\d+)\.", submod_name) # TODO Generalize for other models. This search string is Pythia-specific.
    resid_match = re.search(r"layers\.(\d+)$", submod_name)
    if layer_match:
        submod_layer = int(layer_match.group(1))
    elif resid_match:
        submod_layer = int(resid_match.group(1))
    else:
        raise ValueError(f"No layer number found in submodule name: {submod_name}")
    
    if "attention" in submod_name:
        submod_type = "attn"
    elif "mlp" in submod_name:
        submod_type = "mlp"
    elif len(submod_name.split(".")) == 4:
        submod_type = "resid"
    else:
        raise ValueError(f"No submodule type found in submodule name: {submod_name}")
    
    return submod_layer, submod_type


def load_dictionary(model, submodule_layer, submodule_object, submodule_type, dict_cfg):
        dict_id = "5" # if submodule_type != "attn" else "1"
        dict_path = os.path.join(dict_cfg.dir,
                                 f"{submodule_type}_out_layer{submodule_layer}",
                                 f"{dict_id}_{dict_cfg.size}/ae.pt")
        try:
            submodule_width = submodule_object.out_features
        except AttributeError:
            # is residual. need to load model to get this
            with model.invoke("test") as invoker:
                hidden_states = submodule_object.output.save()
            hidden_states = hidden_states.value
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
            submodule_width = hidden_states.shape[2]
        autoencoder = AutoEncoder(submodule_width, dict_cfg.size).cuda()
        # TODO: add support for both of these cases to the `load_state_dict` method
        try:
            autoencoder.load_state_dict(t.load(dict_path))
        except TypeError:
            autoencoder.load_state_dict(t.load(dict_path).state_dict())
        return autoencoder


def load_submodule_and_dictionary(model, submod_name, dict_cfg: DictionaryCfg):
    submod_layer, submod_type = submodule_name_to_type_layer(submod_name)
    submodule = load_submodule(model, submod_name)
    dictionary = load_dictionary(model, submod_layer, submodule, submod_type, dict_cfg)
    return submodule, dictionary
