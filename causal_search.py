import zstandard as zstd
import json
import os
import io
import random
import argparse
from tqdm import tqdm
from nnsight import LanguageModel
from dictionary_learning.buffer import ActivationBuffer
from dictionary_learning.dictionary import AutoEncoder
from dictionary_learning.training import trainSAE
from datasets import load_dataset
from einops import rearrange
import torch as t

def load_examples(dataset, num_examples, model, seed=12):
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
                            "clean_answer": clean_answer.item(), "patch_answer": patch_answer.item()}
            examples.append(example_dict)
            if len(examples) >= num_examples:
                break
        return examples


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


def get_forward_and_backward_caches_neurons(model, local_submodule, example_dict):
    # corrupted forward pass
    with model.invoke(example_dict["patch_prefix"],
                      fwd_args = {"inference": False}) as invoker_patch:
        patch_submodule_saved = submodule.output.save()
    
    backward_cache = {}
    def backward_hook(module, grad_input, grad_output):
        backward_cache["this"] = grad_output[0]

    hook = model.local_model.gpt_neox.layers[3].mlp.dense_4h_to_h.register_full_backward_hook(backward_hook)

    # clean forward passes
    with model.invoke(example_dict["clean_prefix"],
                      fwd_args = {"inference": False}) as invoker_clean:
        clean_submodule_saved = submodule.output.save()
    
    # clean backward pass
    # clean_submodule_saved.value.retain_grad()
    logits = invoker_clean.output.logits
    metric = compare_probs(logits, example_dict)
    metric.backward()

    return {"clean_activations": clean_submodule_saved.value,
            "patch_activations": patch_submodule_saved.value,
            "clean_gradients":   backward_cache["this"]}


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


# TODO: test
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
    
    # effects, feature_idx = search_submodule_for_phenomenon(model, model.gpt_neox.layers[3].mlp.dense_4h_to_h,
    #                                                       args.dataset)
    top_effects, top_idxs = t.topk(indirect_effects, 10)
    bottom_effects, bottom_idxs = t.topk(indirect_effects, 10, largest=False)
    for effect, idx in zip(top_effects, top_idxs):
        print(f"Top Feature {idx}: {effect:.5f}")
    for effect, idx in zip(bottom_effects, bottom_idxs):
        print(f"Bottom Feature {idx}: {effect:.5f}")
    # for effect, idx in zip(effects, feature_idx):
    #     print(f"Feature {idx}: {effect:.5f}")