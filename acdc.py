import random
import torch as t
from torch import nn
from tqdm import tqdm
from attribution import patching_effect, EffectOut

def consolidated_patching_on(dataset, model, submodules, dictionaries, metric_fn, method='all-folded', steps=10):
    clean_inputs = t.cat([example['clean_prefix'] for example in dataset], dim=0)
    patch_inputs = t.cat([example['patch_prefix'] for example in dataset], dim=0)

    effects, total_effect = patching_effect(
        clean_inputs,
        patch_inputs,
        model,
        submodules,
        dictionaries,
        metric_fn,
        method=method,
        steps=steps,
    )

    return EffectOut(
        effects={k : v.mean(dim=0) for k, v in effects.items()},
        total_effect=total_effect.mean(dim=0),
    )

def patching_on_y(dataset, model, submodules, dictionaries, method='all-folded', steps=10):
    clean_answer_idxs = t.Tensor([example['clean_answer'] for example in dataset]).long()
    patch_answer_idxs = t.Tensor([example['patch_answer'] for example in dataset]).long()

    def metric_fn(model):
        logits = model.embed_out.output[:, -1, :]
        logit_diff = t.gather(
            logits, dim=-1, index=patch_answer_idxs.unsqueeze(-1)
        ) - t.gather(
            logits, dim=-1, index=clean_answer_idxs.unsqueeze(-1)
        )
        return logit_diff.squeeze(-1)

    return consolidated_patching_on(dataset, model, submodules, dictionaries, metric_fn, method, steps)

def patching_on_downstream_feature(dataset, model, submodules, dictionaries,
                                   downstream_submodule, downstream_dictionary,
                                   downstream_feature_id=None, method='all-folded', steps=10):
    def metric_fn(model):
        x = downstream_submodule.output
        f = downstream_dictionary.encode(x)

        if downstream_feature_id:
            f = f[:, :, downstream_feature_id]
        return f.sum(dim=-1)

    return consolidated_patching_on(dataset, model, submodules, dictionaries, metric_fn, method, steps)

# Outdated patching_on_y; split up into two functions now
'''
def patching_on_y(
    dataset,
    model,
    submodules,
    dictionaries,
    method='all-folded',
    steps=10,
):
    clean_inputs = t.cat([example['clean_prefix'] for example in dataset], dim=0)
    patch_inputs = t.cat([example['patch_prefix'] for example in dataset], dim=0)
    clean_answer_idxs = t.Tensor([example['clean_answer'] for example in dataset]).long() # shape [n_examples]
    patch_answer_idxs = t.Tensor([example['patch_answer'] for example in dataset]).long()

    def metric_fn(model):
        logits = model.embed_out.output[:, -1, :] # shape [n_examples, vocab_size]
        logit_diff = t.gather(
            logits,
            dim=-1,
            index=patch_answer_idxs.unsqueeze(-1)
        ) - t.gather(
            logits,
            dim=-1,
            index=clean_answer_idxs.unsqueeze(-1)
        )
        return logit_diff.squeeze(-1) # shape [n_examples]

    effects, total_effect = patching_effect(
        clean_inputs,
        patch_inputs,
        model,
        submodules,
        dictionaries,
        metric_fn,
        method=method,
        steps=steps,
    )

    return EffectOut(
        effects={k : v.mean(dim=0) for k, v in effects.items()}, # v shape: [pos, d_sae]
        total_effect=total_effect.mean(dim=0),
    )
'''

# Outdated patching_on_feature_activation()
'''
def get_hook(submodule_idx, patch, clean, threshold, mean_effects):
    def hook(grad):
        effects = grad * (patch.value - clean.value)
        mean_effects[submodule_idx] += effects[0, -1]
        print("grad", t.topk(grad, 5))
        print("patch acts", t.topk(patch.value, 5))
        print("clean acts", t.topk(clean.value, 5))
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

def patching_on_feature_activation(
        dataset,
        model,
        submodules_lower,
        submodule_upper,
        autoencoders_lower,
        autoencoder_upper,
        upper_feat_idx,
        approx=True,
        threshold=0.2,
        dataset_proportion=1.0,
):
    mean_effects = []   # allow variable-size autoencoders
    for ae in autoencoders_lower:
        mean_effects.append(t.zeros(ae.encoder.out_features))

    num_examples = int(dataset_proportion * len(dataset))
    if num_examples != len(dataset):
        examples = random.sample(dataset, num_examples)
    else:
        examples = dataset

    for example in tqdm(examples, desc="Attribution patching examples", leave=False, total=len(dataset)):
        clean_input, clean_answer_idx = example["clean_prefix"], example["clean_answer"]
        patch_input, patch_answer_idx = example["patch_prefix"], example["patch_answer"]

        patched_features = []
        with model.invoke(patch_input) as invoker:
            for submodule_lower, ae_lower in zip(submodules_lower, autoencoders_lower):
                hidden_states = submodule_lower.output
                if len(hidden_states) > 1:
                    hidden_states = hidden_states[0]
                f = ae_lower.encode(hidden_states)
                patched_features.append(f.save())
                if len(submodule_lower.output) > 1:
                    submodule_lower.output[0] = ae_lower.decode(f)
                else:
                    submodule_lower.output = ae_lower.decode(f)
                
            hidden_states_upper = submodule_upper.output
            if len(hidden_states_upper) > 1:
                hidden_states_upper = hidden_states_upper[0]
            f_upper = autoencoder_upper.encode(hidden_states_upper)
            f_upper.save()
            if len(submodule_upper.output) > 1:
                submodule_upper.output[0] = autoencoder_upper.decode(f)
            else:
                submodule_upper.output = autoencoder_upper.decode(f)

        patch_activation_upper = f_upper.value[0, -1, upper_feat_idx]

        if approx:
            clean_features = []
            with model.invoke(clean_input, fwd_args={'inference' : False}) as invoker:
                for i, (submodule_lower, ae_lower) in enumerate(zip(submodules_lower, autoencoders_lower)):
                    hidden_states_lower = submodule_lower.output
                    if len(hidden_states_lower) > 1:
                        hidden_states_lower = hidden_states_lower[0]
                    f_lower = ae_lower.encode(hidden_states_lower)
                    clean_features.append(f_lower.save())
                    
                    patch, clean = patched_features[i], clean_features[i]
                    
                    hook = get_hook(0, patch, clean, threshold, mean_effects)
                    f.register_hook(hook)

                    if len(submodule_lower.output) > 1:
                        submodule_lower.output[0] = ae_lower.decode(f_lower)
                    else:
                        submodule_lower.output = ae_lower.decode(f_lower)
                
                hidden_states_upper = submodule_upper.output
                if len(hidden_states_upper) > 1:
                    hidden_states_upper = hidden_states_upper[0]
                f_upper = autoencoder_upper.encode(hidden_states_upper)
                f_upper.save()
                if len(submodule_upper.output) > 1:
                    submodule_upper.output[0] = autoencoder_upper.decode(f)
                else:
                    submodule_upper.output = autoencoder_upper.decode(f)

            clean_activations_upper = f_upper.value[0, -1, upper_feat_idx]
            activations_diff = patch_activation_upper - clean_activations_upper
            activations_diff.backward()
            # print(f'Total change: {patch_logit_diff.item() - clean_logit_diff.item()}')

        else: # normal activation patching
            raise NotImplementedError("Activation patching between each feature pair will probably be too slow.")
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
    
    mean_effects = [t.divide(sum_effects, len(examples)) for sum_effects in mean_effects]
    return mean_effects
'''
