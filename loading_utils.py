import os
import re
import json
import random
import torch as t
import torch.nn.functional as F
from nnsight import LanguageModel
from dictionary_learning.dictionary import AutoEncoder
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class DictionaryCfg(): # TODO Move to dictionary_learning repo?
    def __init__(
        self,
        dictionary_dir = '/share/projects/dictionary_circuits/autoencoders/pythia-70m-deduped/',
        dictionary_size = 512 * 64,
        dict_id_attn_out = 0,
        dict_id_mlp_out = 1,
        dict_id_resid_out = 1,
        ) -> None:
        self.dir = dictionary_dir
        self.size = dictionary_size
        self.dict_id_attn_out = dict_id_attn_out
        self.dict_id_mlp_out = dict_id_mlp_out
        self.dict_id_resid_out = dict_id_resid_out


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
            # left padding: reverse, right-pad, reverse
            clean_prefix = t.flip(F.pad(t.flip(clean_prefix, (1,)), (0, pad_length), value=model.tokenizer.pad_token_id), (1,))
            patch_prefix = t.flip(F.pad(t.flip(patch_prefix, (1,)), (0, pad_length), value=model.tokenizer.pad_token_id), (1,))
            # clean_prefix = F.pad(clean_prefix, (0, pad_length), value=model.tokenizer.pad_token_id)
            # patch_prefix = F.pad(patch_prefix, (0, pad_length), value=model.tokenizer.pad_token_id)
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
        # right padding
        # clean_prefix = F.pad(clean_prefix, (0, pad_length), value=model.tokenizer.pad_token_id)
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
    return eval(submodule_str)

    if "." not in submodule_str:
        return getattr(model, submodule_str)
    
    submodules = submodule_str.split(".")
    curr_module = None
    for module in submodules:
        if module == "model":
            continue
        if curr_module is None:
            curr_module = getattr(model, module)
            continue
        curr_module = getattr(curr_module, module)
    return curr_module


def submodule_type_to_name(submodule_type):
    if submodule_type == "mlp":
        return "model.gpt_neox.layers[{}].mlp"
    elif submodule_type == "attn":
        return "model.gpt_neox.layers[{}].attention"
    elif submodule_type.startswith("resid"):
        return "model.gpt_neox.layers[{}]"
    raise ValueError("Unrecognized submodule type. Please select from {mlp, attn, resid}")

def submodule_name_to_type(submod_name):
    if "attention" in submod_name:
        submod_type = "attn"
    elif "mlp" in submod_name:
        submod_type = "mlp"
    elif len(submod_name.split(".")) == 4:
        submod_type = "resid"
    else:
        raise ValueError(f"No submodule type found in submodule name: {submod_name}")
    return submod_type


def submodule_name_to_type_layer(submod_name):
    layer_match = re.search(r"layers\.(\d+)\.", submod_name) # TODO Generalize for other models. This search string is Pythia-specific.
    resid_match = re.search(r"layers\.(\d+)$", submod_name)
    if layer_match:
        submod_layer = int(layer_match.group(1))
    elif resid_match:
        submod_layer = int(resid_match.group(1))
    else:
        raise ValueError(f"No layer number found in submodule name: {submod_name}")
    
    submod_type = submodule_name_to_type(submod_name)
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
            with model.trace("test"), t.inference_mode():
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


def load_submodule_and_dictionary(model: LanguageModel, submod_name: str, dict_cfg: DictionaryCfg):
        submod_layer, submod_type = submodule_name_to_type_layer(submod_name)
        submodule = load_submodule(model, submod_name)
        dictionary = load_dictionary(model, submod_layer, submodule, submod_type, dict_cfg)
        return submodule, dictionary

def load_submodules_and_dictionaries_from_generic(model: LanguageModel, submod_names_generic: list, dict_cfg: DictionaryCfg):
    num_layers = model.config.num_hidden_layers
    all_submodule_names, all_submodules, all_dictionaries = [], [], []
    for layer in range(num_layers):
        submodule_names_layer, submodules_layer, dictionaries_layer = [], [], []
        for submodule_name in submod_names_generic:
            submodule_name = submodule_name.format(str(layer))
            submodule, dictionary = load_submodule_and_dictionary(model, submodule_name, dict_cfg)
            submodule_names_layer.append(submodule_name)
            submodules_layer.append(submodule)
            dictionaries_layer.append(dictionary)
        all_submodule_names.append(submodule_names_layer)
        all_submodules.append(submodules_layer)
        all_dictionaries.append(dictionaries_layer)
    return all_submodule_names, all_submodules, all_dictionaries

def load_submodules_from_generic(model: LanguageModel, submod_names_generic: list):
    num_layers = model.config.num_hidden_layers
    all_submodules = []
    for layer in range(num_layers):
        submodules_layer = []
        for submodule_name in submod_names_generic:
            submodule_name = submodule_name.format(str(layer))
            submodule = load_submodule(model, submodule_name)
            submodules_layer.append(submodule)
        all_submodules.append(submodules_layer)
    return all_submodules