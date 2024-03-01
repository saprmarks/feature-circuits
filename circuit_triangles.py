from dictionary_learning import AutoEncoder
from attribution import EffectOut
import torch as t
from nnsight import LanguageModel
from attribution import patching_effect, jvp
from einops import rearrange
from activation_utils import SparseAct

def get_circuit(
        clean,
        patch,
        model,
        attns,
        mlps,
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

    def unflatten(tensor):
        b, s, f = effects[resids[0]].act.shape
        unflattened = rearrange(tensor, '(b s x) -> b s x', b=b, s=s)
        return SparseAct(act=unflattened[...,:f], res=unflattened[...,f:])
    
    features_by_submod = {
        submod : (effects[submod].to_tensor().flatten().abs() > node_threshold).nonzero().flatten().tolist() for submod in all_submods
    }

    n_layers = len(model.gpt_neox.layers)

    nodes = {'y' : total_effect}
    for i in range(n_layers):
        nodes[f'attn_{i}'] = effects[attns[i]]
        nodes[f'mlp_{i}'] = effects[mlps[i]]
        nodes[f'resid_{i}'] = effects[resids[i]]

    edges = {}
    edges[f'resid_{len(resids) - 1}'] = { 'y' : effects[resids[-1]].to_tensor().flatten() }

    def N(upstream, downstream):
        return jvp(
            clean,
            model,
            dictionaries,
            downstream,
            features_by_submod[downstream],
            upstream,
            grads[downstream],
            deltas[upstream],
            return_without_right=True,
        )


    # now we work backward through the model to get the edges
    for layer in reversed(range(len(resids))):
        resid = resids[layer]
        mlp = mlps[layer]
        attn = attns[layer]

        MR_effect, MR_grad = N(mlp, resid)
        AR_effect, AR_grad = N(attn, resid)

        edges[f'mlp_{layer}']  = { f'resid_{layer}' : MR_effect }
        edges[f'attn_{layer}'] = { f'resid_{layer}' : AR_effect }

        if layer > 1:
            prev_resid = resids[layer - 1]

            RM_effect, _ = N(prev_resid, mlp)
            RA_effect, _ = N(prev_resid, attn)

            edges[f'resid_{layer - 1}'] = { f'mlp_{layer}' : RM_effect }
            edges[f'resid_{layer - 1}'] = { f'attn_{layer}' : RA_effect }

            RMR_effect = jvp(
                clean,
                model,
                dictionaries,
                mlp,
                features_by_submod[resid],
                prev_resid,
                {feat_idx : unflatten(MR_grad[feat_idx].to_dense()) for feat_idx in features_by_submod[resid]},
                deltas[prev_resid],
            )
            RAR_effect = jvp(
                clean,
                model,
                dictionaries,
                attn,
                features_by_submod[resid],
                prev_resid,
                {feat_idx : unflatten(AR_grad[feat_idx].to_dense()) for feat_idx in features_by_submod[resid]},
                deltas[prev_resid],
            )
            RR_effect, _ = N(prev_resid, resid)
            edges[f'resid_{layer - 1}'] = { f'resid_{layer}' : RR_effect - RMR_effect - RAR_effect }
    
    return nodes, edges

if __name__ == '__main__':
    device = 'cuda:0'

    model = LanguageModel('EleutherAI/pythia-70m-deduped', device_map=device)

    attns = [layer.attention for layer in model.gpt_neox.layers]
    mlps = [layer.mlp for layer in model.gpt_neox.layers]
    resids = [layer for layer in model.gpt_neox.layers]
    dictionaries = {}
    for i in range(len(model.gpt_neox.layers)):
        ae = AutoEncoder(512, 512 * 64).to(device)
        ae.load_state_dict(t.load(f'/share/projects/dictionary_circuits/autoencoders/pythia-70m-deduped/attn_out_layer{i}/5_32768/ae.pt'))
        dictionaries[attns[i]] = ae

        ae = AutoEncoder(512, 512 * 64).to(device)
        ae.load_state_dict(t.load(f'/share/projects/dictionary_circuits/autoencoders/pythia-70m-deduped/mlp_out_layer{i}/5_32768/ae.pt'))
        dictionaries[mlps[i]] = ae

        ae = AutoEncoder(512, 512 * 64).to(device)
        ae.load_state_dict(t.load(f'/share/projects/dictionary_circuits/autoencoders/pythia-70m-deduped/resid_out_layer{i}/5_32768/ae.pt'))
        dictionaries[resids[i]] = ae
    
    clean = 'The man'
    patch = 'The men'

    clean_idx = model.tokenizer(" is").input_ids[-1]
    patch_idx = model.tokenizer(" are").input_ids[-1]

    def metric_fn(model):
        return model.embed_out.output[:,-1,patch_idx] - model.embed_out.output[:,-1,clean_idx]
    
    get_circuit(
        clean,
        patch,
        model,
        attns,
        mlps,
        resids,
        dictionaries,
        metric_fn
    )


        
        




