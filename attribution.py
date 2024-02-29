from collections import namedtuple
import torch as t
from tqdm import tqdm
from tensordict import TensorDict
from activation_utils import SparseAct

EffectOut = namedtuple('EffectOut', ['effects', 'deltas', 'grads', 'total_effect'])

def _pe_attrib_all_folded(
        clean,
        patch,
        model,
        submodules,
        dictionaries,
        metric_fn,
):
    hidden_states_clean = {}
    grads = {}
    with model.invoke(clean, fwd_args={'inference' : False}):
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            x = submodule.output
            is_resid = (type(x.shape) == tuple)
            if is_resid:
                x = x[0]
            x_hat, f = dictionary(x, output_features=True) # x_hat implicitly depends on f
            residual = (x - x_hat).detach()
            x_recon = x_hat + residual
            hidden_states_clean[submodule] = f.save()
            grads[submodule] = f.grad.save()
            if is_resid:
                submodule.output[0][:] = x_recon
            else:
                submodule.output = x_recon
            x.grad = x_recon.grad
        metric_clean = metric_fn(model).save()
    metric_clean.value.sum().backward()

    hidden_states_patch = {}
    with model.invoke(patch):
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            x = submodule.output
            is_resid = (type(x.shape) == tuple)
            if is_resid:
                x = x[0]
            f = dictionary.encode(x)
            hidden_states_patch[submodule] = f.save()
        metric_patch = metric_fn(model).save()
    total_effect = (metric_patch.value - metric_clean.value).detach()

    effects = {}
    deltas = {}
    for submodule in submodules:
        patch_state, clean_state = hidden_states_patch[submodule].value, hidden_states_clean[submodule].value.detach()
        delta = patch_state - clean_state if patch_state is not None else -clean_state
        grad = grads[submodule].value
        effect = delta * grad
        effects[submodule] = effect
        deltas[submodule] = delta
        grads[submodule] = grad
    total_effect = total_effect if total_effect is not None else None
    
    return EffectOut(effects, deltas, grads, total_effect)

def _pe_attrib_all_folded_sparseact(
        clean,
        patch,
        model,
        submodules,
        dictionaries,
        metric_fn,
):
    hidden_states_clean = {}
    grads = {}
    with model.invoke(clean, fwd_args={'inference' : False}):
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            x = submodule.output
            is_resid = (type(x.shape) == tuple)
            if is_resid:
                x = x[0]
            x_hat, f = dictionary(x, output_features=True) # x_hat implicitly depends on f
            residual = x - x_hat
            hidden_states_clean[submodule] = SparseAct(act=f.save(), res=residual.save())
            grads[submodule] = SparseAct(act=f.grad.save(), res=residual.grad.save())
            residual.grad = t.zeros_like(residual)
            x_recon = x_hat + residual
            if is_resid:
                submodule.output[0][:] = x_recon
            else:
                submodule.output = x_recon
            x.grad = x_recon.grad
        metric_clean = metric_fn(model).save()
    metric_clean.value.sum().backward()

    hidden_states_patch = {}
    with model.invoke(patch):
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            x = submodule.output
            is_resid = (type(x.shape) == tuple)
            if is_resid:
                x = x[0]
            x_hat, f = dictionary(x, output_features=True) # x_hat implicitly depends on f
            residual = x - x_hat
            hidden_states_patch[submodule] = SparseAct(act=f.save(), res=residual.save())
        metric_patch = metric_fn(model).save()
    total_effect = (metric_patch.value - metric_clean.value).detach()

    effects = {}
    deltas = {}
    for submodule in submodules:
        patch_state, clean_state = hidden_states_patch[submodule].value(), hidden_states_clean[submodule].value().detach()
        delta = patch_state - clean_state if patch_state is not None else -clean_state
        grad = grads[submodule].value()
        effect = delta * grad
        effects[submodule] = effect
        deltas[submodule] = delta
        grads[submodule] = grad
    total_effect = total_effect if total_effect is not None else None
    
    return EffectOut(effects, deltas, grads, total_effect)
    
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
        for submodule in submodules:
            dictionary = dictionaries[submodule]
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
            for submodule in submodules:
                dictionary = dictionaries[submodule]
                x = submodule.output
                if is_resids[submodule]:
                    x = x[0]
                f = dictionary.encode(x)
                hidden_states_patch[submodule] = f.save()
            metric_patch = metric_fn(model).save()
        total_effect = (metric_patch.value - metric_clean.value).detach()

    effects = {}
    deltas = {}
    grads = {}
    for submodule in submodules:
        dictionary = dictionaries[submodule]
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
        grad = sum([f.grad for f in fs]) / steps
        delta = (patch_state.value - clean_state.value).detach() if patch_state is not None else -clean_state.value.detach()
        effect = grad * delta
        effects[submodule] = effect
        deltas[submodule] = delta
        grads[submodule] = grad
    total_effect = total_effect if total_effect is not None else None

    return EffectOut(effects, deltas, grads, total_effect)

def _pe_ig_sparseact(
        clean,
        patch,
        model,
        submodules,
        dictionaries,
        metric_fn,
        steps=10,
):

    hidden_states_clean = {}
    is_resids = {}
    with model.invoke(clean):
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            x = submodule.output
            is_resids[submodule] = (type(x.shape) == tuple)
            if is_resids[submodule]:
                x = x[0]
            f = dictionary.encode(x)
            x_hat = dictionary.decode(f)
            residual = x - x_hat
            hidden_states_clean[submodule] = SparseAct(act=f.save(), res=residual.save())
        metric_clean = metric_fn(model).save()

    hidden_states_patch = {}
    if patch is None:
        hidden_states_patch = {
            k : SparseAct(act=t.zeros_like(v.act), res=t.zeros_like(v.res)) for k, v in hidden_states_clean.items()
        }
        total_effect = None
    else:
        with model.invoke(patch):
            for submodule in submodules:
                dictionary = dictionaries[submodule]
                x = submodule.output
                if is_resids[submodule]:
                    x = x[0]
                f = dictionary.encode(x)
                x_hat = dictionary.decode(f)
                residual = x - x_hat
                hidden_states_patch[submodule] = SparseAct(act=f.save(), res=residual.save())
            metric_patch = metric_fn(model).save()
        total_effect = (metric_patch.value - metric_clean.value).detach()

    effects = {}
    deltas = {}
    grads = {}
    for submodule in submodules:
        dictionary = dictionaries[submodule]
        clean_state = hidden_states_clean[submodule].value()
        patch_state = hidden_states_patch[submodule].value()
        # patch_state, clean_state, residual = \
        #     hidden_states_patch[submodule], hidden_states_clean[submodule].act, hidden_states_clean[submodule].residual
        with model.forward(inference=False) as runner:
            metrics = []
            fs = []
            for step in range(steps):
                alpha = step / steps
                f = (1 - alpha) * clean_state + alpha * patch_state
                f.act.requires_grad = True
                f.res.requires_grad = True
                fs.append(f)
                with runner.invoke(clean):
                    if is_resids[submodule]:
                        submodule.output[0][:] = dictionary.decode(f.act) + f.res # clean_state.res instead of f.res makes this exactly same as the non-sparseact version
                    else:
                        submodule.output = dictionary.decode(f.act) + f.res # clean_state.res instead of f.res makes this exactly same as the non-sparseact version
                    metrics.append(metric_fn(model).save())
        metric = sum([m.value for m in metrics])
        metric.sum().backward()
        mean_grad = sum([f.act.grad for f in fs]) / steps
        mean_residual_grad = sum([f.res.grad for f in fs]) / steps
        grad = SparseAct(act=mean_grad, res=mean_residual_grad)
        delta = (patch_state - clean_state).detach() if patch_state is not None else -clean_state.detach()
        effect = grad * delta
        effects[submodule] = effect
        deltas[submodule] = delta
        grads[submodule] = grad
    total_effect = total_effect if total_effect is not None else None

    return EffectOut(effects, deltas, grads, total_effect)

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
        for submodule in submodules:
            dictionary = dictionaries[submodule]
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
            for submodule in submodules:
                dictionary = dictionaries[submodule]
                x = submodule.output
                is_resid = (type(x.shape) == tuple)
                if is_resid:
                    x = x[0]
                f = dictionary.encode(x)
                hidden_states_patch[submodule] = f.save()
            metric_patch = metric_fn(model).save()
        total_effect = metric_patch.value - metric_clean.value

    effects = {}
    deltas = {}
    for submodule in submodules:
        dictionary = dictionaries[submodule]
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
            effect[idx] = (metric.value - metric_clean.value).sum()
        delta = patch_state.value - clean_state.value
        
        effects[submodule] = effect
        deltas[submodule] = delta
    total_effect = total_effect if total_effect is not None else None

    return EffectOut(effects, deltas, None, total_effect)

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
        return _pe_attrib_all_folded_sparseact(clean, patch, model, submodules, dictionaries, metric_fn)
    elif method == 'ig':
        return _pe_ig_sparseact(clean, patch, model, submodules, dictionaries, metric_fn, steps=steps)
    elif method == 'exact':
        return _pe_exact(clean, patch, model, submodules, dictionaries, metric_fn)
    else:
        raise ValueError(f"Unknown method {method}")

def get_grad(clean, 
             patch, 
             model, 
             dictionaries, 
             upstream_submods, 
             downstream_submod, 
             downstream_features, 
             return_idxs=None # dictionary of upstream idxs to return, for each upstream submodule (and, optionally, each downstream feature idx)
):
    grad = TensorDict()
    with model.invoke(clean, fwd_args={'inference' : False}):
        for upstream_submod in upstream_submods:
            upstream_dictionary = dictionaries[upstream_submod]
            x = upstream_submod.output
            is_resid = (type(x.shape) == tuple)
            if is_resid:
                x = x[0]
            x_hat, f = upstream_dictionary(x, output_features=True)
            grad[upstream_submod] = f.grad.save()
            residual = x - upstream_dictionary(x)
            if is_resid:
                upstream_submod.output[0][:] = x_hat + residual
            else:
                upstream_submod.output = x_hat + residual
        
        y = downstream_submod.output
        downstream_dictionary = dictionaries[downstream_submod]
        is_resid = (type(y.shape) == tuple)
        if is_resid:
            y = y[0]
        f_downstream = downstream_dictionary.encode(y).save()
    
    grads = TensorDict()
    for downstream_feature_idx in downstream_features:
        grads[downstream_feature_idx] = {}
        f_downstream.value[tuple(downstream_feature_idx)].backward(retain_graph=True)
        for upstream_submod in upstream_submods:
            if return_idxs is None or return_idxs[upstream_submod] is None:
                grads[downstream_feature_idx][upstream_submod] = grad[upstream_submod].value
            elif isinstance(return_idxs[upstream_submod], list):
                grads[downstream_feature_idx][upstream_submod] = TensorDict({
                    idx : grad[upstream_submod].value[tuple(idx)] for idx in return_idxs[upstream_submod]
                })
            elif isinstance(return_idxs[upstream_submod], TensorDict):
                grads[downstream_feature_idx][upstream_submod] = TensorDict({
                    idx : grad[upstream_submod].value[tuple(idx)] for idx in return_idxs[upstream_submod][downstream_feature_idx]
                })
    return grads


if __name__ == "__main__":
    from nnsight import LanguageModel
    from dictionary_learning import AutoEncoder

    model = LanguageModel('EleutherAI/pythia-70m-deduped', device_map='cuda:0')
    submodules = []
    submodule_names = {}
    dictionaries = {}
    for layer in range(len(model.gpt_neox.layers)):
        submodule = model.gpt_neox.layers[layer].mlp
        submodule_names[submodule] = f'mlp{layer}'
        submodules.append(submodule)
        ae = AutoEncoder(512, 64 * 512).cuda()
        ae.load_state_dict(t.load(f'/share/projects/dictionary_circuits/autoencoders/pythia-70m-deduped/mlp_out_layer{layer}/5_32768/ae.pt'))
        dictionaries[submodule] = ae

        submodule = model.gpt_neox.layers[layer]
        submodule_names[submodule] = f'resid{layer}'
        submodules.append(submodule)
        ae = AutoEncoder(512, 64 * 512).cuda()
        ae.load_state_dict(t.load(f'/share/projects/dictionary_circuits/autoencoders/pythia-70m-deduped/resid_out_layer{layer}/5_32768/ae.pt'))
        dictionaries[submodule] = ae

    clean_context = ["The man"] #, "The tall boy"]
    patch_context = ["The men"] #, "The tall boys"]
    clean_idx = model.tokenizer(" is").input_ids[-1]
    patch_idx = model.tokenizer(" are").input_ids[-1]

    def metric_fn(model):
        return model.embed_out.output[:,-1,patch_idx] - model.embed_out.output[:,-1,clean_idx]
    
    def compare_effect_outs(eo1, eo2_sparseact):
        for k in ['effects', 'deltas', 'grads']:
            for submod in getattr(eo1, k):
                tensor1 = getattr(eo1, k)[submod]
                tensor2_sparseact = getattr(eo2_sparseact, k)[submod].act
                if not t.allclose(tensor1, tensor2_sparseact):
                    print(f"{k} differs at submod {submod}")
                    print(tensor1.sum())
                    print(tensor2_sparseact.sum())
                    return False
        return True

    # Check that the sparseact version of the function returns the same result as the original

    ## All-folded feature activation test
    effect_out_all_folded = _pe_attrib_all_folded(
        clean_context,
        patch_context,
        model,
        submodules,
        dictionaries,
        metric_fn,
    )
    effect_out_all_folded_sparseact = _pe_attrib_all_folded_sparseact(
        clean_context,
        patch_context,
        model,
        submodules,
        dictionaries,
        metric_fn,
    )
    if compare_effect_outs(effect_out_all_folded, effect_out_all_folded_sparseact):
      print("All-folded test passed")

    ## IG feature activation test
    effect_out_ig = _pe_ig(
        clean_context,
        patch_context,
        model,
        submodules,
        dictionaries,
        metric_fn,
    )
    effect_out_ig_sparseact = _pe_ig_sparseact(
        clean_context,
        patch_context,
        model,
        submodules,
        dictionaries,
        metric_fn,
    )
    if compare_effect_outs(effect_out_ig, effect_out_ig_sparseact):
        print("IG test passed")