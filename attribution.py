import random
from collections import namedtuple
import torch as t
from torch import nn
from tqdm import tqdm

EffectOut = namedtuple('EffectOut', ['effects', 'total_effect'])

def _pe_attrib_all_folded(
        clean,
        patch,
        model,
        submodules,
        dictionaries,
        metric_fn,
):
    with model.invoke(clean, fwd_args={'inference' : False}) as invoker:
        hidden_states_clean = {}
        for submodule, dictionary in zip(submodules, dictionaries):
            is_resid = False
            x = submodule.output
            if len(x[0].shape) > 2:
                is_resid = True
                f = dictionary.encode(x[0])
            else:
                f = dictionary.encode(x)
            f.retain_grad()
            hidden_states_clean[submodule] = f.save()

            x_hat = dictionary.decode(f)
            if is_resid:
                residual = (x[0] - x_hat).detach()
                submodule.output[0] = x_hat + residual
            else:
                residual = (x - x_hat).detach()
                submodule.output = x_hat + residual
        metric_clean = metric_fn(model).save()
    metric_clean.value.sum().backward()

    with model.invoke(patch):
        hidden_states_patch = {}
        for submodule, dictionary in zip(submodules, dictionaries):
            is_resid = False
            x = submodule.output
            if len(x[0].shape) > 2:
                is_resid = True
                f = dictionary.encode(x[0])
            else:
                f = dictionary.encode(x)
            hidden_states_patch[submodule] = f.save()

            x_hat = dictionary.decode(f)
            if is_resid:
                residual = (x[0] - x_hat).detach()
                submodule.output[0] = x_hat + residual
            else:
                residual = (x - x_hat).detach()
                submodule.output = x_hat + residual
        metric_patch = metric_fn(model).save()
    
    total_effect = (metric_patch.value - metric_clean.value) / metric_clean.value

    effects = {}
    for submodule in submodules:
        patch_state, clean_state = hidden_states_patch[submodule], hidden_states_clean[submodule]
        effects[submodule] = ((patch_state.value - clean_state.value) * clean_state.value.grad) / metric_clean.value[:, None, None]

    return EffectOut(effects, total_effect)

def _pe_attrib_separate(
        clean,
        patch,
        model,
        submodules,
        dictionaries,
        metric_fn,
):
    hidden_states_clean = {}
    for submodule, dictionary in zip(submodules, dictionaries):
        with model.invoke(clean, fwd_args={'inference' : False}) as invoker:
            is_resid = False
            x = submodule.output
            if len(x[0].shape) > 2:
                is_resid = True
                f = dictionary.encode(x[0])
            else:
                f = dictionary.encode(x)
            f.retain_grad()
            hidden_states_clean[submodule] = f.save()
            
            x_hat = dictionary.decode(f)
            if is_resid:
                residual = (x[0] - x_hat).detach()
                submodule.output[0] = x_hat + residual
            else:
                residual = (x - x_hat).detach()
                submodule.output = x_hat + residual
            metric_clean = metric_fn(model).save()
        metric_clean.value.sum().backward()

    hidden_states_patch = {}
    with model.invoke(patch):
        for submodule, dictionary in zip(submodules, dictionaries):
            x = submodule.output
            if len(x[0].shape) > 2:
                f = dictionary.encode(x[0])
            else:
                f = dictionary.encode(x)
            hidden_states_patch[submodule] = f.save()
        metric_patch = metric_fn(model).save()

    total_effect = metric_patch.value - metric_clean.value
    
    effects = {}
    for submodule in submodules:
        patch_state, clean_state = hidden_states_patch[submodule], hidden_states_clean[submodule]
        effects[submodule] = ((patch_state.value - clean_state.value) * clean_state.value.grad) / metric_clean.value[:, None, None]

    return EffectOut(effects, total_effect)

def _pe_ig(
        clean,
        patch,
        model,
        submodules,
        dictionaries,
        metric_fn,
        steps=10,
):

    hidden_states_clean = {}
    residuals = {}
    with model.invoke(clean):
        for submodule, dictionary in zip(submodules, dictionaries):
            x = submodule.output
            f = dictionary.encode(x)
            hidden_states_clean[submodule] = f.save()
            x_hat = dictionary.decode(f)
            residuals[submodule] = (x - x_hat).save()
        metric_clean = metric_fn(model).save()

    hidden_states_patch = {}
    with model.invoke(patch):
        for submodule, dictionary in zip(submodules, dictionaries):
            x = submodule.output
            f = dictionary.encode(x)
            hidden_states_patch[submodule] = f.save()
        metric_patch = metric_fn(model).save()

    total_effect = metric_patch.value - metric_clean.value

    effects = {}
    for submodule, dictionary in zip(submodules, dictionaries):
        patch_state, clean_state, residual = \
            hidden_states_patch[submodule], hidden_states_clean[submodule], residuals[submodule]
        with model.forward(inference=False) as runner:
            metrics = []
            fs = []
            for step in range(steps):
                alpha = step / steps
                f = (1 - alpha) * clean_state.value + alpha * patch_state.value
                f.requires_grad = True
                fs.append(f)
                with runner.invoke(clean):
                    submodule.output = dictionary.decode(f) + residual.value
                    metrics.append(metric_fn(model).save())
        metric = sum([m.value for m in metrics])
        metric.sum().backward()
        grad = sum([f.grad for f in fs])
        effects[submodule] = grad * (patch_state.value - clean_state.value) / steps

    return EffectOut(effects, total_effect)

def _pe_exact(
        clean,
        patch,
        model,
        submodules,
        dictionaries,
        metric_fn,
):
    
    hidden_states_clean = {}
    residuals = {}
    with model.invoke(clean) as invoker:
        for submodule, dictionary in zip(submodules, dictionaries):
            x = submodule.output
            f = dictionary.encode(x)
            hidden_states_clean[submodule] = f.save()
            x_hat = dictionary.decode(f)
            residuals[submodule] = (x - x_hat).save()
        metric_clean = metric_fn(model).save()

    hidden_states_patch = {}
    with model.invoke(patch):
        for submodule, dictionary in zip(submodules, dictionaries):
            x = submodule.output
            f = dictionary.encode(x)
            hidden_states_patch[submodule] = f.save()
        metric_patch = metric_fn(model).save()

    total_effect = metric_patch.value - metric_clean.value

    effects = {}
    for submodule, dictionary in zip(submodules, dictionaries):
        patch_state, clean_state, residual = \
            hidden_states_patch[submodule], hidden_states_clean[submodule], residuals[submodule]
        effect = t.zeros_like(clean_state.value)
        
        # iterate over positions and features for which clean and patch differ
        idxs = t.nonzero(patch_state.value - clean_state.value)
        for idx in tqdm(idxs):
            with model.invoke(clean):
                f = clean_state.value.clone()
                f[tuple(idx)] = patch_state.value[tuple(idx)]
                x_hat = dictionary.decode(f)
                submodule.output = x_hat + residual.value
                metric = metric_fn(model).save()
            effect[tuple(idx)] = (metric.value - metric_clean.value).sum()
        
        effects[submodule] = effect

    return EffectOut(effects, total_effect)

def patching_effect(
        clean,
        patch,
        model,
        submodules,
        dictionaries,
        metric_fn,
        method='all-folded',
        steps=10,
):
    if method == 'all-folded':
        return _pe_attrib_all_folded(clean, patch, model, submodules, dictionaries, metric_fn)
    elif method == 'separate':
        return _pe_attrib_separate(clean, patch, model, submodules, dictionaries, metric_fn)
    elif method == 'ig':
        return _pe_ig(clean, patch, model, submodules, dictionaries, metric_fn, steps=steps)
    elif method == 'exact':
        return _pe_exact(clean, patch, model, submodules, dictionaries, metric_fn)
    else:
        raise ValueError(f"Unknown method {method}")