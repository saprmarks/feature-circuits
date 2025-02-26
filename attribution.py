from collections import namedtuple
import torch as t
from tqdm import tqdm
from numpy import ndindex
from loading_utils import Submodule
from activation_utils import SparseAct
from nnsight.envoy import Envoy
from dictionary_learning.dictionary import Dictionary
from typing import Callable

EffectOut = namedtuple('EffectOut', ['effects', 'deltas', 'grads', 'total_effect'])


def _pe_attrib(
        clean,
        patch,
        model,
        submodules: list[Submodule],
        dictionaries: dict[Submodule, Dictionary],
        metric_fn,
        metric_kwargs=dict(),
):
    hidden_states_clean = {}
    grads = {}
    with model.trace(clean):
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            x = submodule.get_activation()
            x_hat, f = dictionary(x, output_features=True) # x_hat implicitly depends on f
            residual = x - x_hat
            hidden_states_clean[submodule] = SparseAct(act=f, res=residual).save()
            grads[submodule] = hidden_states_clean[submodule].grad.save()
            residual.grad = t.zeros_like(residual)
            x_recon = x_hat + residual
            submodule.set_activation(x_recon)
            x.grad = x_recon.grad
        metric_clean = metric_fn(model, **metric_kwargs).save()
        metric_clean.sum().backward()
    hidden_states_clean = {k : v.value for k, v in hidden_states_clean.items()}
    grads = {k : v.value for k, v in grads.items()}

    if patch is None:
        hidden_states_patch = {
            k : SparseAct(act=t.zeros_like(v.act), res=t.zeros_like(v.res)) for k, v in hidden_states_clean.items()
        }
        total_effect = None
    else:
        hidden_states_patch = {}
        with t.no_grad(), model.trace(patch):
            for submodule in submodules:
                dictionary = dictionaries[submodule]
                x = submodule.get_activation()
                x_hat, f = dictionary(x, output_features=True)
                residual = x - x_hat
                hidden_states_patch[submodule] = SparseAct(act=f, res=residual).save()
            metric_patch = metric_fn(model, **metric_kwargs).save()
        total_effect = (metric_patch.value - metric_clean.value).detach()
        hidden_states_patch = {k : v.value for k, v in hidden_states_patch.items()}

    effects = {}
    deltas = {}
    for submodule in submodules:
        patch_state, clean_state, grad = hidden_states_patch[submodule], hidden_states_clean[submodule], grads[submodule]
        delta = patch_state - clean_state.detach() if patch_state is not None else -clean_state.detach()
        effect = delta @ grad
        effects[submodule] = effect
        deltas[submodule] = delta
        grads[submodule] = grad
    total_effect = total_effect if total_effect is not None else None
    
    return EffectOut(effects, deltas, grads, total_effect)


def _pe_ig(
    clean,
    patch,
    model,
    submodules: list[Submodule],
    dictionaries: dict[Submodule, Dictionary],
    metric_fn,
    steps=10,
    metric_kwargs=dict(),
):
    hidden_states_clean = {}
    with t.no_grad(), model.trace(clean):
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            x = submodule.get_activation()
            f = dictionary.encode(x)
            x_hat = dictionary.decode(f)
            residual = x - x_hat
            hidden_states_clean[submodule] = SparseAct(act=f.save(), res=residual.save()) # type: ignore
        metric_clean = metric_fn(model, **metric_kwargs).save()
    hidden_states_clean = {k : v.value for k, v in hidden_states_clean.items()}

    if patch is None:
        hidden_states_patch = {
            k : SparseAct(act=t.zeros_like(v.act), res=t.zeros_like(v.res)) for k, v in hidden_states_clean.items()
        }
        total_effect = None
    else:
        hidden_states_patch = {}
        with t.no_grad(), model.trace(patch):
            for submodule in submodules:
                dictionary = dictionaries[submodule]
                x = submodule.get_activation()
                f = dictionary.encode(x)
                x_hat = dictionary.decode(f)
                residual = x - x_hat
                hidden_states_patch[submodule] = SparseAct(act=f.save(), res=residual.save()) # type: ignore
            metric_patch = metric_fn(model, **metric_kwargs).save()
        total_effect = (metric_patch.value - metric_clean.value).detach()
        hidden_states_patch = {k : v.value for k, v in hidden_states_patch.items()}

    effects = {}
    deltas = {}
    grads = {}
    for submodule in submodules:
        dictionary = dictionaries[submodule]
        clean_state = hidden_states_clean[submodule]
        patch_state = hidden_states_patch[submodule]
        with model.trace() as tracer:
            metrics = []
            fs = []
            for step in range(steps):
                alpha = step / steps
                f = (1 - alpha) * clean_state + alpha * patch_state
                f.act.requires_grad_().retain_grad()
                f.res.requires_grad_().retain_grad()
                fs.append(f)
                with tracer.invoke(clean):
                    submodule.set_activation(dictionary.decode(f.act) + f.res)
                    metrics.append(metric_fn(model, **metric_kwargs))
            metric = sum([m for m in metrics])
            metric.sum().backward()
        
        mean_grad = sum([f.act.grad for f in fs]) / steps
        mean_residual_grad = sum([f.res.grad for f in fs]) / steps
        grad = SparseAct(act=mean_grad, res=mean_residual_grad) # type: ignore
        delta = (patch_state - clean_state).detach() if patch_state is not None else -clean_state.detach()
        effect = grad @ delta

        effects[submodule] = effect
        deltas[submodule] = delta
        grads[submodule] = grad

    return EffectOut(effects, deltas, grads, total_effect)


def _pe_exact(
    clean,
    patch,
    model,
    submodules: list[Submodule],
    dictionaries: dict[Submodule, Dictionary],
    metric_fn,
):
    hidden_states_clean = {}
    with t.no_grad(), model.trace(clean):
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            x = submodule.get_activation()
            f = dictionary.encode(x)
            x_hat = dictionary.decode(f)
            residual = x - x_hat
            hidden_states_clean[submodule] = SparseAct(act=f, res=residual).save() # type: ignore
        metric_clean = metric_fn(model).save()
    hidden_states_clean = {k : v.value for k, v in hidden_states_clean.items()}

    if patch is None:
        hidden_states_patch = {
            k : SparseAct(act=t.zeros_like(v.act), res=t.zeros_like(v.res)) for k, v in hidden_states_clean.items()
        }
        total_effect = None
    else:
        hidden_states_patch = {}
        with t.no_grad(), model.trace(patch):
            for submodule in submodules:
                dictionary = dictionaries[submodule]
                x = submodule.get_activation()
                f = dictionary.encode(x)
                x_hat = dictionary.decode(f)
                residual = x - x_hat
                hidden_states_patch[submodule] = SparseAct(act=f, res=residual).save() # type: ignore
            metric_patch = metric_fn(model).save()
        total_effect = metric_patch.value - metric_clean.value
        hidden_states_patch = {k : v.value for k, v in hidden_states_patch.items()}

    effects = {}
    deltas = {}
    for submodule in submodules:
        dictionary = dictionaries[submodule]
        clean_state = hidden_states_clean[submodule]
        patch_state = hidden_states_patch[submodule]
        effect = SparseAct(act=t.zeros_like(clean_state.act), resc=t.zeros(*clean_state.res.shape[:-1])).to(model.device)
        
        # iterate over positions and features for which clean and patch differ
        idxs = t.nonzero(patch_state.act - clean_state.act)
        for idx in tqdm(idxs):
            with t.no_grad(), model.trace(clean):
                f = clean_state.act.clone()
                f[tuple(idx)] = patch_state.act[tuple(idx)]
                x_hat = dictionary.decode(f)
                submodule.set_activation(x_hat + clean_state.res)
                metric = metric_fn(model).save()
            effect.act[tuple(idx)] = (metric.value - metric_clean.value).sum()

        for idx in list(ndindex(effect.resc.shape)): # type: ignore
            with t.no_grad(), model.trace(clean):
                res = clean_state.res.clone()
                res[tuple(idx)] = patch_state.res[tuple(idx)] # type: ignore
                x_hat = dictionary.decode(clean_state.act)
                submodule.set_activation(x_hat + res)
                metric = metric_fn(model).save()
            effect.resc[tuple(idx)] = (metric.value - metric_clean.value).sum() # type: ignore
        
        effects[submodule] = effect
        deltas[submodule] = patch_state - clean_state
    total_effect = total_effect if total_effect is not None else None

    return EffectOut(effects, deltas, None, total_effect)

def patching_effect(
        clean,
        patch,
        model,
        submodules: list[Submodule],
        dictionaries: dict[Submodule, Dictionary],
        metric_fn: Callable[[Envoy], t.Tensor],
        method='attrib',
        steps=10,
        metric_kwargs=dict()
):

    if method == 'attrib':
        return _pe_attrib(clean, patch, model, submodules, dictionaries, metric_fn, metric_kwargs=metric_kwargs)
    elif method == 'ig':
        return _pe_ig(clean, patch, model, submodules, dictionaries, metric_fn, steps=steps, metric_kwargs=metric_kwargs)
    elif method == 'exact':
        return _pe_exact(clean, patch, model, submodules, dictionaries, metric_fn)
    else:
        raise ValueError(f"Unknown method {method}")

def jvp(
    input,
    model,
    dictionaries,
    downstream_submod,
    downstream_features,
    upstream_submod,
    left_vec: SparseAct,
    right_vec: SparseAct,
    intermediate_stopgrads: list[Submodule] = [],
):
    downstream_dict, upstream_dict = dictionaries[downstream_submod], dictionaries[upstream_submod]
    b, s, n_feats = downstream_features.act.shape

    if t.all(downstream_features.to_tensor() == 0):
        return t.sparse_coo_tensor(
            t.zeros((2 * downstream_features.act.dim(), 0), dtype=t.long), 
            t.zeros(0), 
            size=(b, s, n_feats+1, b, s, n_feats+1)
        ).to(model.device)

    vjv_values = {}

    downstream_feature_idxs = downstream_features.to_tensor().nonzero()
    with model.trace(input):
        # forward pass modifications
        x = upstream_submod.get_activation()
        x_hat, f = upstream_dict.hacked_forward_for_sfc(x) # hacking around an nnsight bug
        x_res = x - x_hat
        upstream_submod.set_activation(x_hat + x_res)
        upstream_act = SparseAct(act=f, res=x_res).save()
        
        y = downstream_submod.get_activation()
        y_hat, g = downstream_dict.hacked_forward_for_sfc(y) # hacking around an nnsight bug
        y_res = y - y_hat
        downstream_act = SparseAct(act=g, res=y_res)

        to_backprops = (left_vec @ downstream_act).to_tensor()

        for downstream_feat_idx in downstream_feature_idxs:
            # stop grad
            for submodule in intermediate_stopgrads:
                submodule.stop_grad()
            x_res.grad = t.zeros_like(x_res.grad)

            vjv = (upstream_act.grad @ right_vec).to_tensor()
            to_backprops[tuple(downstream_feat_idx)].backward(retain_graph=True)
            vjv_values[downstream_feat_idx] = vjv.save() # type: ignore

    vjv_indices = t.stack(list(vjv_values.keys()), dim=0).T
    vjv_values = t.stack([v.value for v in vjv_values.values()], dim=0)

    return t.sparse_coo_tensor(vjv_indices, vjv_values, size=(b, s, n_feats+1, b, s, n_feats+1))