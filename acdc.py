import random
import torch as t
from torch import nn
from tqdm import tqdm
from attribution import patching_effect, EffectOut
from loading_utils import load_submodule_and_dictionary

def consolidated_patching_on(dataset, model, upstream_submodule_names, dict_cfg, metric_fn, method='separate', steps=10):
    clean_inputs = t.cat([example['clean_prefix'] for example in dataset], dim=0)
    patch_inputs = t.cat([example['patch_prefix'] for example in dataset], dim=0)

    effects, total_effect = patching_effect(
        clean_inputs,
        patch_inputs,
        model,
        upstream_submodule_names,
        dict_cfg,
        metric_fn,
        method=method,
        steps=steps,
    )

    return EffectOut(
        effects={k : v.mean(dim=0) for k, v in effects.items()},
        total_effect=total_effect.mean(dim=0),
    )

def patching_on_y(dataset, model, upstream_submodule_names, dict_cfg, method='separate', steps=10):
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

    return consolidated_patching_on(dataset, model, upstream_submodule_names, dict_cfg, metric_fn, method, steps)

def patching_on_downstream_feature(
    dataset, 
    model, 
    upstream_submodule_names,
    downstream_submodule_name, 
    downstream_feature_id,
    dict_cfg,
    method='separate', 
    steps=10):
    downstream_submodule, downstream_dictionary = load_submodule_and_dictionary(model, downstream_submodule_name, dict_cfg)
    def metric_fn(model):
        x = downstream_submodule.output
        if len(x[0].shape) > 2:
            f = downstream_dictionary.encode(x[0])
        else:
            f = downstream_dictionary.encode(x)

        if downstream_feature_id:
            f = f[:, :, downstream_feature_id]
        return f.sum(dim=-1)

    return consolidated_patching_on(dataset, model, upstream_submodule_names, dict_cfg, metric_fn, method, steps)
