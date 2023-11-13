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

def load_examples(dataset, num_examples, seed=12):
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


def get_activation_caches(model, submodule, autoencoder,
                           clean_input, patch_input):
    # 1 clean forward, 1 patch forward
    with model.generate(max_new_tokens=1, pad_token_id=model.tokenizer.pad_token_id) as generator:
        with generator.invoke(clean_input, scan=False) as invoker:
            clean_hidden_states = submodule.output.save()
    clean_dictionary_activations = autoencoder.encode(clean_hidden_states.value.to("cuda"))
    with model.generate(max_new_tokens=1, pad_token_id=model.tokenizer.pad_token_id) as generator:
        with generator.invoke(patch_input, scan=False) as invoker:
            patch_hidden_states = submodule.output.save()
    patch_dictionary_activations = autoencoder.encode(patch_hidden_states.value.to("cuda"))
    return {"clean_dictionary_activations": clean_dictionary_activations,
            "patch_dictionary_activations": patch_dictionary_activations}


def search_for_phenomenon(model, submodule, autoencoder, dataset,
                          num_return_features=10,
                          num_examples=1):
    examples = load_examples(dataset, num_examples)
    print(f"Number of valid examples: {len(examples)}")
    
    num_features = autoencoder.dict_size
    indirect_effects = t.zeros(len(examples), num_features)
    for example_idx, example in tqdm(enumerate(examples), desc="Example", total=len(examples)):
        dictionary_activations = get_activation_caches(model, submodule, autoencoder,
                                        example["clean_prefix"], example["patch_prefix"])

        # get clean logits
        with model.forward(example["clean_prefix"]) as invoker:
            acts = dictionary_activations["clean_dictionary_activations"]
            clean_reconstructed_activations = autoencoder.decode(acts)
            submodule.output = clean_reconstructed_activations
        logits = invoker.output.logits
        logits = logits[:,-1,:]
        clean_probs = logits.softmax(dim=-1)
        y_clean = clean_probs[0][example["patch_answer"]].item() / clean_probs[0][example["clean_answer"]].item()
        
        # get logits with single patched feature
        for feature_idx in tqdm(range(0, num_features), leave=False, desc="Feature"):
            with model.forward(example["clean_prefix"]) as invoker:
                # patch single feature in dictionary
                acts = dictionary_activations["clean_dictionary_activations"]
                acts[:, -1, feature_idx] = dictionary_activations["patch_dictionary_activations"][:, -1, feature_idx]
                patch_reconstructed_activations = autoencoder.decode(acts)
                submodule.output = patch_reconstructed_activations
            logits = invoker.output.logits
            logits = logits[:,-1,:]
            patch_probs = logits.softmax(dim=-1)
            y_patch = patch_probs[0][example["patch_answer"]].item() / patch_probs[0][example["clean_answer"]].item()
            indirect_effects[example_idx, feature_idx] = (y_patch - y_clean) / y_clean
    
    # take mean across examples
    indirect_effects = t.mean(indirect_effects, dim=0)
    # returns list of tuples of type (indirect_effect, feature_index)
    return t.topk(indirect_effects, num_return_features)

if __name__ == "__main__":
    model = LanguageModel("EleutherAI/pythia-70m-deduped")
    submodule = model.gpt_neox.layers[3].mlp.dense_4h_to_h
    autoencoder = AutoEncoder(512, 2048).cuda()
    autoencoder.load_state_dict(t.load('autoencoders/ae_mlp3_c4_lr0.0001_resample25000_dict2048.pt').state_dict())
    autoencoder = autoencoder.to("cuda")
    
    effects, feature_idx = search_for_phenomenon(model,
                                                 model.gpt_neox.layers[3].mlp.dense_4h_to_h,
                                                 autoencoder,
                                                 "phenomena/vocab/simple.json")
    for effect, idx in zip(effects, feature_idx):
        print(f"Feature {idx}: {effect:.5f}")