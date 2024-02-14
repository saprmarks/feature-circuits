from collections import namedtuple
import torch as t
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
    # get clean states
    hidden_states_clean = {}
    with model.invoke(clean, fwd_args={'inference' : False}) as invoker:
        for submodule, dictionary in zip(submodules, dictionaries):
            x = submodule.output
            is_resid = (type(x.shape) == tuple)
            if is_resid:
                x = x[0]
            f = dictionary.encode(x)
            f.retain_grad()
            hidden_states_clean[submodule] = f.save()

            x_hat = dictionary.decode(f)
            residual = (x - x_hat).detach()
            if is_resid:
                submodule.output[0][:] = x_hat + residual
            else:
                submodule.output = x_hat + residual
        metric_clean = metric_fn(model).save()

    # backprop to get gradients for features
    metric_clean.value.sum().backward()
    
    if patch is None:
        hidden_states_patch = {
            k : t.zeros_like(v) for k, v in hidden_states_clean.items()
        }
        total_effect = None
    else:
        hidden_states_patch = {}
        with model.invoke(patch):
            for submodule, dictionary in zip(submodules, dictionaries):
                x = submodule.output
                is_resid = (type(x.shape) == tuple)
                if is_resid:
                    x = x[0]
                f = dictionary.encode(x)
                hidden_states_patch[submodule] = f.save()
            metric_patch = metric_fn(model).save()
        total_effect = metric_patch.value - metric_clean.value

    effects = {}
    for submodule in submodules:
        patch_state, clean_state = hidden_states_patch[submodule], hidden_states_clean[submodule]
        effects[submodule] = (patch_state.value - clean_state.value).detach() * clean_state.value.grad

    return EffectOut(effects, total_effect)

def _pe_attrib_separate(
        clean,
        patch,
        model,
        submodules,
        dictionaries,
        metric_fn,
        grad_y_wrt_downstream=1,
):
    hidden_states_clean = {}

    for submodule, dictionary in zip(submodules, dictionaries):
        with model.invoke(clean, fwd_args={'inference' : False}):
            x = submodule.output
            is_resid = (type(x.shape) == tuple)
            if is_resid:
                x = x[0]
            f = dictionary.encode(x)
            f.retain_grad()
            hidden_states_clean[submodule] = f.save()
            
            x_hat = dictionary.decode(f)
            residual = (x - x_hat).detach()
            if is_resid:
                submodule.output[0][:] = x_hat + residual
            else:
                submodule.output = x_hat + residual
            metric_clean = metric_fn(model).save()
        metric_clean.value.sum().backward()

    if patch is None:
        hidden_states_patch = {
            k : t.zeros_like(v) for k, v in hidden_states_clean.items()
        }
        total_effect = None
    else:
        hidden_states_patch = {}
        with model.invoke(patch):
            for submodule, dictionary in zip(submodules, dictionaries):
                x = submodule.output
                is_resid = (type(x.shape) == tuple)
                if is_resid:
                    x = x[0]
                f = dictionary.encode(x)
                hidden_states_patch[submodule] = f.save()
            metric_patch = metric_fn(model).save()
        total_effect = metric_patch.value - metric_clean.value

    effects = {}
    grads_y_wrt_us_features = {}
    for submodule in submodules:
        patch_state, clean_state = hidden_states_patch[submodule], hidden_states_clean[submodule]
        effects[submodule] = (patch_state.value - clean_state.value).detach() * clean_state.value.grad * grad_y_wrt_downstream
        grads_y_wrt_us_features[submodule] = clean_state.value.grad * grad_y_wrt_downstream 
    return EffectOut(effects, total_effect), grads_y_wrt_us_features

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
    is_resids = {}
    with model.invoke(clean):
        for submodule, dictionary in zip(submodules, dictionaries):
            x = submodule.output
            is_resids[submodule] = (type(x.shape) == tuple)
            if is_resids[submodule]:
                x = x[0]
            f = dictionary.encode(x)
            hidden_states_clean[submodule] = f.save()
            x_hat = dictionary.decode(f)
            residuals[submodule] = (x - x_hat).save()
        metric_clean = metric_fn(model).save()

    hidden_states_patch = {}
    if patch is None:
        hidden_states_patch = {
            k : t.zeros_like(v) for k, v in hidden_states_clean.items()
        }
        total_effect = None
    else:
        with model.invoke(patch):
            for submodule, dictionary in zip(submodules, dictionaries):
                x = submodule.output
                if is_resids[submodule]:
                    x = x[0]
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
                    if is_resids[submodule]:
                        submodule.output[0][:] = dictionary.decode(f) + residual.value
                    else:
                        submodule.output = dictionary.decode(f) + residual
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
    with model.invoke(clean):
        for submodule, dictionary in zip(submodules, dictionaries):
            x = submodule.output
            is_resid = (type(x.shape) == tuple)
            if is_resid:
                x = x[0]
            f = dictionary.encode(x)
            hidden_states_clean[submodule] = f.save()
            x_hat = dictionary.decode(f)
            residuals[submodule] = (x - x_hat).save()
        metric_clean = metric_fn(model).save()

    if patch is None:
        hidden_states_patch = {
            k : t.zeros_like(v) for k, v in hidden_states_clean.items()
        }
        total_effect = None
    else:
        hidden_states_patch = {}
        with model.invoke(patch):
            for submodule, dictionary in zip(submodules, dictionaries):
                x = submodule.output
                is_resid = (type(x.shape) == tuple)
                if is_resid:
                    x = x[0]
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
                if is_resid:
                    submodule.output[0][:] = x_hat + residual.value
                else:
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
        grad_y_wrt_downstream=None, # currently only used in circuit discovery, thus only implemented for separate method
):
    if method == 'all-folded':
        return _pe_attrib_all_folded(clean, patch, model, submodules, dictionaries, metric_fn)
    elif method == 'separate':
        return _pe_attrib_separate(clean, patch, model, submodules, dictionaries, metric_fn, grad_y_wrt_downstream)
    elif method == 'ig':
        return _pe_ig(clean, patch, model, submodules, dictionaries, metric_fn, steps=steps)
    elif method == 'exact':
        return _pe_exact(clean, patch, model, submodules, dictionaries, metric_fn)
    else:
        raise ValueError(f"Unknown method {method}")