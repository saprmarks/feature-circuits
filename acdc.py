# %%

import torch as t
from torch import nn
from tqdm import tqdm

def get_hook(submodule_idx, patch, clean, threshold, mean_effects):
    def hook(grad):
        effects = grad * (patch.value - clean.value)
        mean_effects[submodule_idx] += effects[0, -1]
        effects = t.where(
            t.gt(grad * (patch.value - clean.value), threshold),
            grad,
            t.zeros_like(grad)
        )

        """
        # print(f"Submodule {submodule_idx}")
        for feature_idx in t.nonzero(effects):
            value = effects[tuple(feature_idx)] * (patch.value - clean.value)[tuple(feature_idx)]
            # print(f"Multindex: {tuple(feature_idx.tolist())}, Value: {value}")
        # print()
        """

        return effects
    return hook

def patching_on_y(
        dataset,
        model,
        submodules,
        autoencoders,
        approx=True,
        threshold=0.2,
):
    mean_effects = []   # allow variable-size autoencoders
    for ae in autoencoders:
        mean_effects.append(t.zeros(ae.encoder.out_features))

    for example in tqdm(dataset, desc="Examples", leave=False, total=len(dataset)):
        clean_input, clean_answer_idx = example["clean_prefix"], example["clean_answer"]
        patch_input, patch_answer_idx = example["patch_prefix"], example["patch_answer"]

        patched_features = []
        with model.invoke(patch_input) as invoker:
            for submodule, ae in zip(submodules, autoencoders):
                hidden_states = submodule.output
                if len(hidden_states) > 1:
                    hidden_states = hidden_states[0]
                f = ae.encode(hidden_states)
                patched_features.append(f.save())
                if len(submodule.output) > 1:
                    submodule.output[0] = ae.decode(f)
                else:
                    submodule.output = ae.decode(f)
        logits = invoker.output.logits
        patch_logit_diff = logits[0, -1, patch_answer_idx] - logits[0, -1, clean_answer_idx]

        if approx:
            clean_features = []
            with model.invoke(clean_input, fwd_args={'inference' : False}) as invoker:
                for i, (submodule, ae) in enumerate(zip(submodules, autoencoders)):
                    hidden_states = submodule.output
                    if len(hidden_states) > 1:
                        hidden_states = hidden_states[0]
                    f = ae.encode(hidden_states)
                    clean_features.append(f.save())
                    
                    patch, clean = patched_features[i], clean_features[i]
                    hook = get_hook(i, patch, clean, threshold, mean_effects)
                    f.register_hook(hook)

                    if len(submodule.output) > 1:
                        submodule.output[0] = ae.decode(f)
                    else:
                        submodule.output = ae.decode(f)
                
            logits = invoker.output.logits
            clean_logit_diff = logits[0, -1, patch_answer_idx] - logits[0, -1, clean_answer_idx]
            clean_logit_diff.backward()
            # print(f'Total change: {patch_logit_diff.item() - clean_logit_diff.item()}')

        else: # normal activation patching
            # get logits on clean run
            with model.invoke(clean_input) as invoker:
                pass
            logits = invoker.output.logits
            clean_logit_diff = logits[0, -1, patch_answer_idx] - logits[0, -1, clean_answer_idx]

            print(f'Clean diff: {clean_logit_diff.item()}')
            print(f'Patch diff: {patch_logit_diff.item()}')

            for i, (submodule, ae, patch) in tqdm(enumerate(zip(submodules, autoencoders, patched_features)), position=0, desc="Layer"):
                for feat in tqdm(range(ae.dict_size), position=1, desc="Feature", leave=False):
                    with model.invoke(clean_input) as invoker:
                        f = ae.encode(submodule.output)
                        f[:,:,feat] = patch.value[:,:,feat]
                        submodule.output = ae.decode(f)
                    logits = invoker.output.logits
                    logit_diff = logits[0, -1, patch_answer_idx] - logits[0, -1, clean_answer_idx]
                    if logit_diff - clean_logit_diff > threshold:
                        print(f"Layer {i}, Feature {feat}, Diff: {logit_diff.item()}")
    
    mean_effects = [t.divide(sum_effects, len(dataset)) for sum_effects in mean_effects]
    return mean_effects

if __name__ == "__main__":
    from nnsight import LanguageModel
    from dictionary_learning.dictionary import AutoEncoder

    model = LanguageModel('EleutherAI/pythia-70m-deduped', device_map='cuda:0')
    layers = len(model.gpt_neox.layers)

    submodules = [
        model.gpt_neox.layers[i].mlp.dense_4h_to_h for i in range(layers)
    ]

    # We'll probably need to replace this; will take a lot of memory in larger models
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

    grads = effect_on_y(clean, patch, model, submodules, autoencoders, threshold=0.5)
    # %%
