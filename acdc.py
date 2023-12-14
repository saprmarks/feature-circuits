# %%

import random
import torch as t
import json
from tqdm import tqdm
from torch import nn

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


def get_hook(layer, patch, clean, threshold):
    def hook(grad):
        effects = grad * (patch.value - clean.value)
        mean_effects[layer] += effects[0, -1]
        effects = t.where(
            t.gt(grad * (patch.value - clean.value), threshold),
            grad,
            t.zeros_like(grad)
        )

        # print(f"Layer {layer}")
        for idx in t.nonzero(effects):
            value = grad[tuple(idx)] * (patch.value - clean.value)[tuple(idx)]
            # print(f"Multindex: {tuple(idx.tolist())}, Value: {value}")
        # print()

        return effects
    return hook


def find_circuit(
        dataset,
        model,
        submodules,
        autoencoders,
        threshold=0.5,
):
    for example in tqdm(dataset, desc="Example:", total=len(dataset)):
        patched_features = []
        with model.invoke(example["patch_prefix"]) as invoker:
            for submodule, ae in zip(submodules, autoencoders):
                f = ae.encode(submodule.output)
                patched_features.append(f.save())
                submodule.output = ae.decode(f)
        logits = invoker.output.logits
        patch_logit_diff = logits[0, -1, example["patch_answer"]] - logits[0, -1, example["clean_answer"]]

        clean_features = []
        with model.invoke(example["clean_prefix"], fwd_args={'inference': False}) as invoker:
            for i, (submodule, ae) in enumerate(zip(submodules, autoencoders)):
                f = ae.encode(submodule.output)
                clean_features.append(f.save())
                
                patch, clean = patched_features[i], clean_features[i]
                hook = get_hook(i, patch, clean, threshold)
                f.register_hook(hook)

                submodule.output = ae.decode(f)
        logits = invoker.output.logits
        clean_logit_diff = logits[0, -1, example["patch_answer"]] - logits[0, -1, example["clean_answer"]]

        clean_logit_diff.backward()
        # print(f'Total change: {patch_logit_diff.item() - clean_logit_diff.item()}')


# %%
from nnsight import LanguageModel
from dictionary_learning.dictionary import AutoEncoder

model = LanguageModel('EleutherAI/pythia-70m-deduped', device_map='cuda:0')
layers = len(model.gpt_neox.layers)
dataset = load_examples("phenomena/vocab/nounpp.json", 10, model)

submodules = [
    model.gpt_neox.layers[i].mlp.dense_4h_to_h for i in range(layers)
]

autoencoders = []
for i in range(layers):
    ae = AutoEncoder(512, 16 * 512)
    ae.load_state_dict(t.load(f'/share/projects/dictionary_circuits/autoencoders/pythia-70m-deduped/mlp_out_layer{i}/0_8192/ae.pt'))
    autoencoders.append(ae)

clean = (
    "The men", " are"
)
patch = (
    "The man", " is"
)

mean_effects = t.zeros((6, 8192)).to("cuda:0")
grads = find_circuit(dataset, model, submodules, autoencoders, threshold=0.5)
mean_effects /= 10

for layer in range(mean_effects.shape[0]):
    effects = t.where(
        mean_effects[layer] > 0.01,
        mean_effects[layer],
        t.zeros_like(mean_effects[layer])
    )

    print(f"Layer {layer}")
    for idx in t.nonzero(effects):
        layer_idx = (layer, idx[-1])
        value = effects[tuple(idx)]
        print(f"Index: {layer_idx}. Value: {value}")
    print()
# %%
