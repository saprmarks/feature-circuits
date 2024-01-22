#%%
# Setup
import random
import json
from nnsight import LanguageModel
from torch.nn import functional as F
import torch as t
import sys
import os
sys.path.append("../")
from loading_utils import load_examples, load_submodule_and_dictionary, DictionaryCfg
import argparse
from circuit import Circuit

model_name = "EleutherAI/pythia-70m-deduped"
submodules_generic = "model.gpt_neox.layers.{}.mlp.dense_4h_to_h"
dictionary_dir = "/share/projects/dictionary_circuits/autoencoders/pythia-70m-deduped"
dictionary_size = 32768
dataset_dir_share = "/share/projects/dictionary_circuits/data/phenomena/simple.json"
num_examples = 100
patch_method = "separate"

eff_outs = []
for pad_length in [None, 3]:
    print(f"\n\nPad length: {pad_length}\n\n")
    if "," in submodules_generic:
        submodules = submodules_generic.split(",")
    else:
        submodules = [submodules_generic]

    model = LanguageModel(model_name, dispatch=True)
    model.local_model.requires_grad_(True)
    dataset = load_examples(dataset_dir_share, num_examples, model, pad_to_length=pad_length)

    circuit = Circuit(model, submodules, dictionary_dir, dictionary_size, dataset)
    eff_outs.append(circuit.locate_circuit(patch_method=patch_method))
    print(circuit.to_dict())

# %%
print(f'total effect difference for padding: {eff_outs[0].total_effect.abs().sum() - eff_outs[1].total_effect.abs().sum()}')
for i in range (6):
    print(f"\nSubmodule: {i}")
    submod_name = f'model.gpt_neox.layers.{i}.mlp.dense_4h_to_h'
    effects_pos1_nopad = eff_outs[0].effects[submod_name].abs()[1, :]
    effects_pos1_pad = eff_outs[1].effects[submod_name].abs()[1, :]
    print(f'sum of effect diff: {effects_pos1_pad.sum() - effects_pos1_nopad.sum()}')
    print(f'max effect diff: {(effects_pos1_nopad - effects_pos1_pad).max()}')


# %%
# model = LanguageModel('EleutherAI/pythia-70m-deduped', dispatch=True)

# dictionary_dir = "/share/projects/dictionary_circuits/autoencoders/pythia-70m-deduped"
# dictionary_size = 32768
# submodule, dictionary = load_submodule_and_dictionary(
#     model, 
#     submod_name='model.gpt_neox.layers.0.mlp.dense_4h_to_h',
#     dict_cfg=DictionaryCfg(dictionary_dir, dictionary_size)
# )


# prompt = "The man"
# pad_length = 3
# # Inference False

# unpadded = model.tokenizer(prompt, return_tensors="pt", padding=False).input_ids
# padded = F.pad(unpadded, (0, pad_length), value=model.tokenizer.pad_token_id)

# print(f'padded tensor: {padded}')
# print(f'unpadded tensor: {unpadded}')

# # %%
# # Run forward pass with and without padding
# with model.invoke(padded, fwd_args={'inference' : False}) as invoker:
#     is_resid = False
#     mlp_act_pad = submodule.output
#     if len(mlp_act_pad[0].shape) > 2:
#         is_resid = True
#         f = dictionary.encode(mlp_act_pad[0])
#     else:
#         f = dictionary.encode(mlp_act_pad)
#     f.retain_grad()
#     mlp_act_pad = mlp_act_pad.save()
#     dict_act_pad = f.save()

# with model.invoke(unpadded, fwd_args={'inference' : False}) as invoker:
#     is_resid = False
#     mlp_act_unpad = submodule.output
#     if len(mlp_act_unpad[0].shape) > 2:
#         is_resid = True
#         f = dictionary.encode(mlp_act_unpad[0])
#     else:
#         f = dictionary.encode(mlp_act_unpad)
#     f.retain_grad()
#     mlp_act_unpad = mlp_act_unpad.save()
#     dict_act_unpad = f.save()

# # %%
# # Mlp activations
# mlp_abs_diff = (mlp_act_pad.value[0, 1, :] - mlp_act_unpad.value[0, 1, :]).sum()
# rel_diff = mlp_abs_diff / mlp_act_unpad.value[0, 1, :].sum()
# print(f'shapes: {mlp_act_pad.value.shape}, {mlp_act_unpad.value.shape}')
# print(f'abs diff: {mlp_abs_diff}')
# print(f'rel diff: {rel_diff}')

# # %%
# # Dict activations
# dict_abs_diff = (dict_act_pad.value[0, 1, :] - dict_act_unpad.value[0, 1, :]).sum()
# rel_diff = dict_abs_diff / dict_act_unpad.value[0, 1, :].sum()
# print(f'shapes: {dict_act_pad.value.shape}, {dict_act_unpad.value.shape}')
# print(f'abs diff: {dict_abs_diff}')
# print(f'rel diff: {rel_diff}')
# # %%