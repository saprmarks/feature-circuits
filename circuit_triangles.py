from dictionary_learning import AutoEncoder
import torch as t
from nnsight import LanguageModel
from attribution import patching_effect, jvp

def get_circuit(
        clean,
        patch,
        model,
        mlps,
        attns,
        resids,
        dictionaries,
        metric_fn,
        node_threshold=0.1,
        edge_threshold=0.01,
):
    all_submods = [submod for layer_submods in zip(mlps, attns, resids) for submod in layer_submods]
    
    # first get the patching effect of everything on y
    effects, deltas, grads, total_effect = patching_effect(
        clean,
        patch,
        model,
        all_submods,
        dictionaries,
        metric_fn
    )

    features_by_submod = {
        submod : (effects[submod].abs() > 0).nonzero() for submod in all_submods
    }

    edges = {}
    # now we work backward through the model to get the edges
    for layer in reversed(range(len(resids))):
        resid = resids[layer]
        edges[f'resid_{layer}'] = { 'y' : effects[resid]}

        mlp = mlps[layer]
        edges[f'mlp_{layer}'] = { f'resid_{layer}' : effects[mlp]}

        attn = attns[layer]
        edges[f'attn_{layer}'] = { f'resid_{layer}' : effects[attn]}

        if layer > 1:
            prev_resid = resids[layer - 1]
            
            mlp_grad = jvp(
                clean,
                patch,
                model,
                mlp,
                prev_resid,
                dictionaries,
                grads[mlp],
            )
            edges[f'resid_{layer - 1}'] = { f'mlp_{layer}' : mlp_grad * deltas[prev_resid] }

            attn_grad = jvp(
                clean,
                patch,
                model,
                attn,
                prev_resid,
                dictionaries,
                grads[attn],
            )
            edges[f'resid_{layer - 1}'] = { f'attn_{layer}' : attn_grad * deltas[prev_resid] }
    
    return features_by_submod, edges

        
        




