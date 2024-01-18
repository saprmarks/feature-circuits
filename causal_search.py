import zstandard as zstd
import json
import os
import io
import random
import argparse
from tqdm import tqdm
from collections import defaultdict
import numpy as np

import torch as t
from nnsight import LanguageModel
from dictionary_learning.buffer import ActivationBuffer
from dictionary_learning.dictionary import AutoEncoder
from dictionary_learning.training import trainSAE
from circuitsvis.activations import text_neuron_activations
from circuitsvis.topk_tokens import topk_tokens
from loading_utils import load_examples, load_submodule
from datasets import load_dataset
from einops import rearrange
from dictionary_learning.search_utils import feature_effect

# Helper functions for causal search
def compare_probs(logits, example_dict):
    last_token_logits = logits[:,-1,:]
    probs = last_token_logits.softmax(dim=-1)
    prob_ratio = t.divide(probs[0][example_dict["clean_answer"]],
                          probs[0][example_dict["patch_answer"]])
    return prob_ratio


def get_dictionary_activation_caches(model, submodule, autoencoder,
                                     example_dict):
    clean_input = example_dict["clean_prefix"]
    patch_input = example_dict["patch_prefix"]

    # 1 clean forward, 1 patch forward
    with model.generate(max_new_tokens=1, pad_token_id=model.tokenizer.pad_token_id) as generator:
        with generator.invoke(clean_input, scan=False) as invoker:
            clean_hidden_states = submodule.output.save()
    clean_dictionary_activations = autoencoder.encode(clean_hidden_states.value.to("cuda"))
    with model.generate(max_new_tokens=1, pad_token_id=model.tokenizer.pad_token_id) as generator:
        with generator.invoke(patch_input, scan=False) as invoker:
            patch_hidden_states = submodule.output.save()
    patch_dictionary_activations = autoencoder.encode(patch_hidden_states.value.to("cuda"))
    return {"clean_activations": clean_dictionary_activations,
            "patch_activations": patch_dictionary_activations}


def get_submodule_activation_caches(model, submodule,
                                    example_dict):
    clean_input = example_dict["clean_prefix"]
    patch_input = example_dict["patch_prefix"]

    # 1 clean forward, 1 patch forward
    with model.generate(max_new_tokens=1, pad_token_id=model.tokenizer.pad_token_id) as generator:
        with generator.invoke(clean_input, scan=False) as invoker:
            clean_hidden_states = submodule.output.save()
    with model.generate(max_new_tokens=1, pad_token_id=model.tokenizer.pad_token_id) as generator:
        with generator.invoke(patch_input, scan=False) as invoker:
            patch_hidden_states = submodule.output.save()
    return {"clean_activations": clean_hidden_states.value,
            "patch_activations": patch_hidden_states.value}


def get_forward_and_backward_caches(model, submodule, autoencoder,
                                    example_dict, metric_fn=compare_probs):
    # corrupted forward pass
    with model.invoke(example_dict["patch_prefix"],
                      fwd_args = {"inference": False}) as invoker_patch:
        x = submodule.output
        f = autoencoder.encode(x)
        patch_f_saved = f.save()
        x_hat = autoencoder.decode(f)
        submodule.output = x_hat
    
    # clean forward passes
    with model.invoke(example_dict["clean_prefix"],
                      fwd_args = {"inference": False}) as invoker_clean:
        x = submodule.output
        f = autoencoder.encode(x)
        clean_f_saved = f.save()
        x_hat = autoencoder.decode(f)
        submodule.output = x_hat
    
    # clean backward pass
    clean_f_saved.value.retain_grad()
    logits = invoker_clean.output.logits
    metric = metric_fn(logits, example_dict)
    metric.backward()

    return {"clean_activations": clean_f_saved.value,
            "patch_activations": patch_f_saved.value,
            "clean_gradients":   clean_f_saved.value.grad}


def get_forward_and_backward_caches_neurons(model, submodule, example_dict,
                                            metric_fn=compare_probs):
    # corrupted forward pass
    with model.invoke(example_dict["patch_prefix"],
                      fwd_args = {"inference": False}) as invoker_patch:
        x = submodule.output
        clean_x_saved = x.save()
    
    # clean forward passes
    with model.invoke(example_dict["clean_prefix"],
                      fwd_args = {"inference": False}) as invoker_clean:
        x = submodule.output
        patch_x_saved = x.save()
    
    # clean backward pass
    clean_x_saved.value.retain_grad()
    logits = invoker_clean.output.logits
    metric = metric_fn(logits, example_dict)
    metric.backward()

    return {"clean_activations": clean_x_saved.value,
            "patch_activations": patch_x_saved.value,
            "clean_gradients":   clean_x_saved.value.grad}


def get_forward_and_backward_caches_wrt_features(model, submodule_lower, submodule_upper,
                                                 autoencoder_lower, autoencoder_upper, feat_idx_upper,
                                                 example_dict):
    # corrupted forward pass
    with model.invoke(example_dict["patch_prefix"],
                      fwd_args = {"inference": False}) as invoker_patch:
        x = submodule_lower.output
        f = autoencoder_lower.encode(x)
        patch_f_saved = f.save()
        x_hat = autoencoder_lower.decode(f)
        submodule_lower.output = x_hat
    
    # clean forward pass
    upper_autoencoder_acts = None
    with model.invoke(example_dict["clean_prefix"],
                      fwd_args = {"inference": False}) as invoker_clean:
        x = submodule_lower.output
        f = autoencoder_lower.encode(x)
        clean_f_saved = f.save()
        x_hat = autoencoder_lower.decode(f)
        submodule_lower.output = x_hat

        y = submodule_upper.output
        g = autoencoder_upper.encode(y)
        clean_g_saved = g.save()
    
    # clean backward pass
    clean_f_saved.value.retain_grad()
    clean_feat_activation = clean_g_saved[:, -1, feat_idx_upper]
    clean_feat_activation.backward()

    return {"clean_activations": clean_f_saved.value,
            "patch_activations": patch_f_saved.value,
            "clean_gradients":   clean_f_saved.value.grad}


def search_dictionary_for_phenomenon(model, submodule, autoencoder, dataset,
                                     num_return_features=10,
                                     num_examples=100):
    examples = load_examples(dataset, num_examples, model)
    print(f"Number of valid examples: {len(examples)}")
    
    num_features = autoencoder.dict_size
    indirect_effects = t.zeros(len(examples), num_features)
    for example_idx, example in tqdm(enumerate(examples), desc="Example", total=len(examples)):
        dictionary_activations = get_dictionary_activation_caches(model, submodule, autoencoder,
                                        example["clean_prefix"], example["patch_prefix"])

        # get clean logits
        with model.forward(example["clean_prefix"]) as invoker_clean:
            acts = dictionary_activations["clean_activations"]
            clean_reconstructed_activations = autoencoder.decode(acts)
            submodule.output = clean_reconstructed_activations
        logits = invoker_clean.output.logits
        logits = logits[:,-1,:]     # logits at final token
        clean_probs = logits.softmax(dim=-1)
        y_clean = clean_probs[0][example["patch_answer"]].item() / clean_probs[0][example["clean_answer"]].item()
        
        # get logits with single patched feature
        for feature_idx in tqdm(range(0, num_features), leave=False, desc="Feature"):
            with model.forward(example["clean_prefix"]) as invoker_patch:
                # patch single feature in dictionary
                acts = dictionary_activations["clean_activations"]
                acts[:, -1, feature_idx] = dictionary_activations["patch_activations"][:, -1, feature_idx]
                patch_reconstructed_activations = autoencoder.decode(acts)
                submodule.output = patch_reconstructed_activations
            logits = invoker_patch.output.logits
            logits = logits[:,-1,:]     # logits at final token
            patch_probs = logits.softmax(dim=-1)
            y_patch = patch_probs[0][example["patch_answer"]].item() / patch_probs[0][example["clean_answer"]].item()
            indirect_effects[example_idx, feature_idx] = (y_patch - y_clean) / y_clean
    
    # take mean across examples
    indirect_effects = t.mean(indirect_effects, dim=0)
    return indirect_effects
    # returns list of tuples of type (indirect_effect, feature_index)
    # return t.topk(indirect_effects, num_return_features)


def search_submodule_for_phenomenon(model, submodule, dataset,
                          num_return_features=10,
                          num_examples=100):
    examples = load_examples(dataset, num_examples, model)
    print(f"Number of valid examples: {len(examples)}")
    
    num_features = submodule.out_features
    indirect_effects = t.zeros(len(examples), num_features)
    for example_idx, example in tqdm(enumerate(examples), desc="Example", total=len(examples)):
        dictionary_activations = get_submodule_activation_caches(model, submodule,
                                        example["clean_prefix"], example["patch_prefix"])

        # get clean logits
        with model.forward(example["clean_prefix"]) as invoker_clean:
            pass    # no interventions necessary
        logits = invoker_clean.output.logits
        logits = logits[:,-1,:]     # logits at final token
        clean_probs = logits.softmax(dim=-1)
        y_clean = clean_probs[0][example["patch_answer"]].item() / clean_probs[0][example["clean_answer"]].item()
        
        # get logits with single patched feature
        for feature_idx in tqdm(range(0, num_features), leave=False, desc="Feature"):
            with model.forward(example["clean_prefix"]) as invoker_patch:
                # patch single feature in dictionary
                acts = submodule.output
                acts[:, -1, feature_idx] = dictionary_activations["patch_activations"][:, -1, feature_idx]
                submodule.output = acts
            logits = invoker_patch.output.logits
            logits = logits[:,-1,:]     # logits at final token
            patch_probs = logits.softmax(dim=-1)
            y_patch = patch_probs[0][example["patch_answer"]].item() / patch_probs[0][example["clean_answer"]].item()
            indirect_effects[example_idx, feature_idx] = (y_patch - y_clean) / y_clean
    
    # take mean across examples
    indirect_effects = t.mean(indirect_effects, dim=0)
    return indirect_effects
    # returns list of tuples of type (indirect_effect, feature_index)
    # return t.topk(indirect_effects, num_return_features)


def attribution_patching(model, submodule, autoencoder, examples, metric_fn=compare_probs):
    num_features = autoencoder.dict_size
    indirect_effects = t.zeros(len(examples), num_features)
    for example_idx, example in tqdm(enumerate(examples), desc="Example", leave=False, total=len(examples)):
        # get forward and backward patches
        activations_gradients = get_forward_and_backward_caches(model, submodule,
                                                                autoencoder, example,
                                                                metric_fn=metric_fn)
        act_clean = activations_gradients["clean_activations"][:,-1]
        act_patch = activations_gradients["patch_activations"][:,-1]
        grad_clean = activations_gradients["clean_gradients"][:,-1]
        indirect_effects[example_idx] = grad_clean * (act_patch - act_clean)
    
    indirect_effects = t.mean(indirect_effects, dim=0)
    return indirect_effects


def attribution_patching_neurons(model, submodule, examples):
    # examples = load_examples(dataset, num_examples, model)
    print(f"Number of valid examples: {len(examples)}")
    
    num_features = submodule.out_features
    indirect_effects = t.zeros(len(examples), num_features)
    for example_idx, example in tqdm(enumerate(examples), desc="Example", total=len(examples)):
        # get forward and backward patches
        activations_gradients = get_forward_and_backward_caches_neurons(model, submodule,
                                                                example)
        act_clean = activations_gradients["clean_activations"][:,-1]
        act_patch = activations_gradients["patch_activations"][:,-1]
        grad_clean = activations_gradients["clean_gradients"][:,-1]
        indirect_effects[example_idx] = grad_clean * (act_patch - act_clean)
    
    indirect_effects = t.mean(indirect_effects, dim=0)
    return indirect_effects


def attribution_patching_wrt_features(model, submodule_lower, submodule_upper,
                                      autoencoder_lower, autoencoder_upper, feat_idx_upper,
                                      examples):
    num_features_lower = autoencoder_lower.dict_size
    indirect_effects = t.zeros(len(examples), num_features_lower)
    for example_idx, example in tqdm(enumerate(examples), desc="Example", leave=False, total=len(examples)):
        # get forward and backward patches
        activations_gradients = get_forward_and_backward_caches_wrt_features(model, submodule_lower, submodule_upper,
                                                                autoencoder_lower, autoencoder_upper, feat_idx_upper,
                                                                example)
        act_clean = activations_gradients["clean_activations"][:,-1]
        act_patch = activations_gradients["patch_activations"][:,-1]
        grad_clean = activations_gradients["clean_gradients"][:,-1]
        indirect_effects[example_idx] = grad_clean * (act_patch - act_clean)
    
    indirect_effects = t.mean(indirect_effects, dim=0)
    return indirect_effects


def examine_dimension(model, submodule, buffer, dictionary=None,
                      dim_idx=None, k=30):
    def _list_decode(x):
        if isinstance(x, int):
            return model.tokenizer.decode(x)
        else:
            return [_list_decode(y) for y in x]
        
    # are we working with residuals?
    is_resid = False
    with model.invoke("dummy text") as invoker:
        if type(submodule.output.shape) == tuple:
            is_resid = True
    
    if dictionary is not None:
        dimensions = dictionary.encoder.out_features
    else:
        dimensions = submodule.output[0].shape[-1] if is_resid else submodule.output.shape[-1]
    
    inputs = buffer.tokenized_batch().to("cuda")
    with model.generate(max_new_tokens=1, pad_token_id=model.tokenizer.pad_token_id) as generator:
        with generator.invoke(inputs['input_ids'], scan=False) as invoker:
            hidden_states = submodule.output.save()
    hidden_states = hidden_states.value[0] if is_resid else hidden_states.value
    if dictionary is not None:
        activations = dictionary.encode(hidden_states)
    else:
        activations = hidden_states

    flattened_acts = rearrange(activations, 'b n d -> (b n) d')
    freqs = (flattened_acts !=0).sum(dim=0) / flattened_acts.shape[0]

    k = k
    if dim_idx is not None:
        feat = dim_idx
    else:
        feat = random.randint(0, dimensions-1)
    acts = activations[:, :, feat].cpu()
    flattened_acts = rearrange(acts, 'b l -> (b l)')
    topk_indices = t.argsort(flattened_acts, dim=0, descending=True)[:k]
    batch_indices = topk_indices // acts.shape[1]
    token_indices = topk_indices % acts.shape[1]
    
    tokens = [
        inputs['input_ids'][batch_idx, :token_idx+1].tolist() for batch_idx, token_idx in zip(batch_indices, token_indices)
    ]
    tokens = _list_decode(tokens)
    activations = [
        acts[batch_idx, :token_id+1, None, None] for batch_idx, token_id in zip(batch_indices, token_indices)
    ]

    token_acts_sums = defaultdict(float)
    token_acts_counts = defaultdict(int)
    token_mean_acts = defaultdict(float)
    for context_idx, context in enumerate(activations):
        for token_idx in range(activations[context_idx].shape[0]):
            token = tokens[context_idx][token_idx]
            activation = activations[context_idx][token_idx].item()
            token_acts_sums[token] += activation
            token_acts_counts[token] += 1
    for token in token_acts_sums:
        token_mean_acts[token] = token_acts_sums[token] / token_acts_counts[token]
    token_mean_acts = {k: v for k, v in sorted(token_mean_acts.items(), key=lambda item: item[1], reverse=True)}
    top_tokens = []
    i = 0
    for token in token_mean_acts:
        top_tokens.append((token, token_mean_acts[token]))
        i += 1
        if i >= 10:
            break

    top_contexts = text_neuron_activations(tokens, activations)

    # this isn't working as expected, for some reason
    """"
    top_affected = []
    for input in inputs["input_ids"]:
        top_affected.append(feature_effect(model, submodule, dictionary, dim_idx, model.tokenizer.decode(input)))
        print(top_affected[-1])
    """

    return {"top_contexts": top_contexts,
            "top_tokens": top_tokens}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--autoencoders", "-a", type=str,
                        default='autoencoders/ae_mlp3_c4_lr0.0001_resample25000_dict32768.pt')
    parser.add_argument("--models", "-m", type=str,
                        default='autoencoders/ae_mlp3_c4_lr0.0001_resample25000_dict32768.pt')
    parser.add_argument("--dataset", "-d", type=str,
                        default="phenomena/vocab/simple.json")
    parser.add_argument("--num_examples", "-n", type=int, default=100,
                        help="Number of example pairs to use in the causal search.")
    parser.add_argument("--submodule", "-s", type=str,
                        default="model.gpt_neox.layers.3.mlp.dense_4h_to_h")
    args = parser.parse_args()

    # load model and specify submodule
    model = LanguageModel("EleutherAI/pythia-70m-deduped", dispatch=True)
    model.local_model.requires_grad_(True)
    submodule = load_submodule(model, args.submodule)
    submodule_width = submodule.out_features
    dataset = load_examples(args.dataset, args.num_examples, model)

    # load autoencoder
    autoencoder_size = 32768
    if "_sz" in args.autoencoder:
        autoencoder_size = int(args.autoencoder.split("_sz")[1].split("_")[0].split(".")[0])
    elif "_dict" in args.autoencoder:
        autoencoder_size = int(args.autoencoder.split("_dict")[1].split("_")[0].split(".")[0])
    autoencoder = AutoEncoder(submodule_width, autoencoder_size).cuda()
    try:
        autoencoder.load_state_dict(t.load(args.autoencoder))
    except TypeError:
        autoencoder.load_state_dict(t.load(args.autoencoder).state_dict())
    autoencoder = autoencoder.to("cuda")
    
    indirect_effects = attribution_patching_wrt_features(model,
                                            submodule,
                                            autoencoder,
                                            dataset)
    
    top_effects, top_idxs = t.topk(indirect_effects, 10)
    bottom_effects, bottom_idxs = t.topk(indirect_effects, 10, largest=False)
    for effect, idx in zip(top_effects, top_idxs):
        print(f"Top Feature {idx}: {effect:.5f}")
    for effect, idx in zip(bottom_effects, bottom_idxs):
        print(f"Bottom Feature {idx}: {effect:.5f}")
