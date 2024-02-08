import torch as t
from attribution import patching_effect, EffectOut

def consolidated_patching_on(dataset, model, upstream_submodules, upstream_dictionaries, metric_fn, grad_y_wrt_downstream, method='separate', steps=10):
    clean_inputs = t.cat([example['clean_prefix'] for example in dataset], dim=0)
    patch_inputs = t.cat([example['patch_prefix'] for example in dataset], dim=0)

    effects, total_effect, grads_y_wrt_us_features = patching_effect(
        clean_inputs,
        patch_inputs,
        model,
        upstream_submodules,
        upstream_dictionaries,
        metric_fn,
        method=method,
        steps=steps,
        grad_y_wrt_downstream=grad_y_wrt_downstream,
    )

    return EffectOut(
        effects={k : v.mean(dim=0) for k, v in effects.items()},
        total_effect=total_effect.mean(dim=0),
    ), grads_y_wrt_us_features

def patching_on_y(dataset, model, submodules, dictionaries, method='separate', steps=10, grad_y_wrt_downstream=1): # grad of y wrt y is 1.
    clean_answer_idxs, patch_answer_idxs, prefix_lengths_wo_pad = [], [], []
    for example in dataset:
        clean_answer_idxs.append(example['clean_answer'])
        patch_answer_idxs.append(example['patch_answer'])
        prefix_lengths_wo_pad.append(example['prefix_length_wo_pad'])
    clean_answer_idxs = t.Tensor(clean_answer_idxs).long()
    patch_answer_idxs = t.Tensor(patch_answer_idxs).long()
    prefix_lengths_wo_pad = t.Tensor(prefix_lengths_wo_pad).int()

    def metric_fn(model):
        indices_first_dim = t.arange(clean_answer_idxs.shape[0])
        logits = model.embed_out.output[indices_first_dim, prefix_lengths_wo_pad-1, :]
        logit_diff = t.gather(
            logits, dim=-1, index=patch_answer_idxs.unsqueeze(-1)
        ) - t.gather(
            logits, dim=-1, index=clean_answer_idxs.unsqueeze(-1)
        )
        return logit_diff.squeeze(-1)

    return consolidated_patching_on(dataset, model, submodules, dictionaries, metric_fn, method, steps, grad_y_wrt_downstream)

def patching_on_downstream_feature(
    dataset, 
    model, 
    upstream_submodules,
    upstream_dictionaries,
    downstream_submodule,
    downstream_dictionary,
    downstream_feature_id,
    grad_y_wrt_downstream,
    method='separate', 
    steps=10,):
    def metric_fn(model):
        x = downstream_submodule.output
        is_resid = (type(x.shape) == tuple)
        if is_resid:
            x = x[0]
        f = downstream_dictionary.encode(x)
        if downstream_feature_id:
            f = f[:, :, downstream_feature_id]
        else:
            f = f.sum(dim=-1)
        return f.sum(dim=-1)

    return consolidated_patching_on(dataset, model, upstream_submodules, upstream_dictionaries, metric_fn, method, steps, grad_y_wrt_downstream)
