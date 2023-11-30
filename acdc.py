# %%

import torch as t
from torch import nn

def get_hook(layer, patch, clean, threshold):
    def hook(grad):
        effects = t.where(
            t.gt(grad * (patch.value - clean.value), threshold),
            grad,
            t.zeros_like(grad)
        )

        print(f"Layer {layer}")
        for idx in t.nonzero(effects):
            value = effects[tuple(idx)] * (patch.value - clean.value)[tuple(idx)]
            print(f"Multindex: {tuple(idx.tolist())}, Value: {value}")
        print()

        return effects
    
    return hook

def find_circuit(
        clean,
        patch,
        model,
        submodules,
        autoencoders,
        threshold=0.5,
):
    clean_input, clean_answer = clean
    patch_input, patch_answer = patch
    clean_answer_idx = model.tokenizer(clean_answer)['input_ids'][0]
    patch_answer_idx = model.tokenizer(patch_answer)['input_ids'][0]

    patched_features = []
    with model.invoke(patch_input) as invoker:
        for submodule, ae in zip(submodules, autoencoders):
            f = ae.encode(submodule.output)
            patched_features.append(f.save())
            submodule.output = ae.decode(f)
    logits = invoker.output.logits
    patch_logit_diff = logits[0, -1, patch_answer_idx] - logits[0, -1, clean_answer_idx]

    clean_features = []
    with model.invoke(clean_input, fwd_args={'inference' : False}) as invoker:
        for i, (submodule, ae) in enumerate(zip(submodules, autoencoders)):
            f = ae.encode(submodule.output)
            clean_features.append(f.save())
            
            patch, clean = patched_features[i], clean_features[i]
            hook = get_hook(i, patch, clean, threshold)
            f.register_hook(hook)

            submodule.output = ae.decode(f)
        
    logits = invoker.output.logits
    clean_logit_diff = logits[0, -1, patch_answer_idx] - logits[0, -1, clean_answer_idx]
    clean_logit_diff.backward()
    print(f'Total change: {patch_logit_diff.item() - clean_logit_diff.item()}')


# %%
from nnsight import LanguageModel
from dictionary_learning.dictionary import AutoEncoder

model = LanguageModel('EleutherAI/pythia-70m-deduped', device_map='cuda:0')
layers = len(model.gpt_neox.layers)

submodules = [
    model.gpt_neox.layers[i].mlp.dense_4h_to_h for i in range(layers)
]

autoencoders = []
for i in range(layers):
    ae = AutoEncoder(512, 16 * 512)
    ae.load_state_dict(t.load(f'/share/projects/dictionary_circuits/autoencoders/pythia-70m-deduped/mlp_out_layer{i}/0_8192/ae.pt'))
    autoencoders.append(ae)

clean = (
    "The man", " is"
)
patch = (
    "The men", " are"
)

grads = find_circuit(clean, patch, model, submodules, autoencoders, threshold=0.5)
# %%
