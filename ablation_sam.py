from nnsight import LanguageModel
import torch as t
from dictionary_learning import AutoEncoder
from argparse import ArgumentParser
from activation_utils import SparseAct

# TODO make work if there are also sequence positions
# for now, assumes that the indices of nodes are integers, rather than, e.g. tuples
def run_with_ablations(
        clean,
        patch, # none if zero ablations
        model, 
        submodules, # list of submodules 
        dictionaries, # dictionaries[submodule] is an autoencoder for submodule's output
        nodes, # nodes[submoduel] is a list of nodes
        metric_fn, # metric_fn(model, **metric_kwargs) -> t.Tensor
        metric_kwargs=dict(),
        complement=False, # if True, then use the complement of nodes
        ablation_fn=lambda x: x, # what to do to the patch hidden states to produce values for ablation, default resample
        handle_resids='default', # or 'remove' to zero ablate all; 'keep' to keep all
    ):

    if patch is None: patch = clean
    patch_states = {}
    with model.trace(patch), t.inference_mode():
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            x = submodule.output
            if type(x.shape) == tuple:
                x = x[0]
            x_hat, f = dictionary(x, output_features=True)
            patch_states[submodule] = SparseAct(act=f, res=x - x_hat).save()
    patch_states = {k : ablation_fn(v.value) for k, v in patch_states.items()}


    with model.trace(clean), t.inference_mode():
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            submod_nodes = nodes[submodule]
            x = submodule.output
            is_tuple = type(x.shape) == tuple
            if is_tuple:
                x = x[0]
            f = dictionary.encode(x)

            # ablate features
            if complement:
                node_idxs = t.zeros(dictionary.dict_size, dtype=t.bool)
            else:
                node_idxs = t.ones(dictionary.dict_size, dtype=t.bool)
            for idx in submod_nodes:
                if not isinstance(idx, str):
                    node_idxs[idx] = not node_idxs[idx]
            f[...,node_idxs] = patch_states[submodule].act[...,node_idxs]

            # handle residuals
            assert handle_resids in ['default', 'remove', 'keep']
            if handle_resids == 'remove':
                res = patch_states[submodule].res
            elif handle_resids == 'keep':
                res = x - dictionary(x)
            elif handle_resids == 'default' and ( ('res' in submod_nodes) ^ (not complement) ):
                res = patch_states[submodule].res
            else:
                res = x - dictionary(x)
            
            if is_tuple:
                submodule.output[0][:] = dictionary.decode(f) + res
            else:
                submodule.output = dictionary.decode(f) + res
        metric = metric_fn(model, **metric_kwargs).save()
    return metric.value


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.1)
    parser.add_argument('--ablation', type=str, default='resample')
    parser.add_argument('--circuit', type=str, default='rc_dict10_node0.01_edge0.001_n30_aggsum.pt')
    parser.add_argument('--faithfulness', action='store_true')
    parser.add_argument('--completeness', action='store_true')
    parser.add_argument('--handle_resids', type=str, default='default')
    args = parser.parse_args()

    model = LanguageModel('EleutherAI/pythia-70m-deduped', device_map='cuda:0', dispatch=True)


    # submodules = \
    #     [layer.attention for layer in model.gpt_neox.layers] + \
    #     [layer.mlp for layer in model.gpt_neox.layers] + \
    #     [layer for layer in model.gpt_neox.layers]

    submodules = [model.gpt_neox.embed_in] + \
        [layer.attention for layer in model.gpt_neox.layers] + \
        [layer.mlp for layer in model.gpt_neox.layers] + \
        [layer for layer in model.gpt_neox.layers]
    dictionaries = {}
    ae = AutoEncoder(512, 64 * 512).to('cuda:0')
    ae.load_state_dict(t.load('/share/projects/dictionary_circuits/autoencoders/pythia-70m-deduped/embed/10_32768/ae.pt'))
    dictionaries[model.gpt_neox.embed_in] = ae
    for i in range(len(model.gpt_neox.layers)):
        ae = AutoEncoder(512, 64 * 512).to('cuda:0')
        ae.load_state_dict(t.load(f'/share/projects/dictionary_circuits/autoencoders/pythia-70m-deduped/attn_out_layer{i}/10_32768/ae.pt'))
        dictionaries[model.gpt_neox.layers[i].attention] = ae

        ae = AutoEncoder(512, 64 * 512).to('cuda:0')
        ae.load_state_dict(t.load(f'/share/projects/dictionary_circuits/autoencoders/pythia-70m-deduped/mlp_out_layer{i}/10_32768/ae.pt'))
        dictionaries[model.gpt_neox.layers[i].mlp] = ae

        ae = AutoEncoder(512, 64 * 512).to('cuda:0')
        ae.load_state_dict(t.load(f'/share/projects/dictionary_circuits/autoencoders/pythia-70m-deduped/resid_out_layer{i}/10_32768/ae.pt'))
        dictionaries[model.gpt_neox.layers[i]] = ae
    
    circuit = t.load(f'circuits/{args.circuit}')

    examples = circuit['examples']

    nodes_out = circuit['nodes']
    nodes = {}
    submod_nodes = (nodes_out['embed'] > args.threshold).nonzero().squeeze(-1)
    nodes[model.gpt_neox.embed_in] = list(submod_nodes.act) + (['res'] if len(submod_nodes.resc) > 0 else [])
    for i in range(len(model.gpt_neox.layers)):
        submod_nodes = (nodes_out[f'attn_{i}'] > args.threshold).nonzero().squeeze(-1)
        nodes[model.gpt_neox.layers[i].attention] = list(submod_nodes.act) + (['res'] if len(submod_nodes.resc) > 0 else [])
        submod_nodes = (nodes_out[f'mlp_{i}'] > args.threshold).nonzero().squeeze(-1)
        nodes[model.gpt_neox.layers[i].mlp] = list(submod_nodes.act) + (['res'] if len(submod_nodes.resc) > 0 else [])
        submod_nodes = (nodes_out[f'resid_{i}'] > args.threshold).nonzero().squeeze(-1)
        nodes[model.gpt_neox.layers[i]] = list(submod_nodes.act) + (['res'] if len(submod_nodes.resc) > 0 else [])

    clean_inputs = t.cat([e['clean_prefix'] for e in examples], dim=0).to('cuda:0')
    clean_answer_idxs = t.tensor([e['clean_answer'] for e in examples], dtype=t.long, device='cuda:0')
    patch_inputs = t.cat([e['patch_prefix'] for e in examples], dim=0).to('cuda:0')
    patch_answer_idxs = t.tensor([e['patch_answer'] for e in examples], dtype=t.long, device='cuda:0')
    def metric_fn(model):
        return (
            - t.gather(model.embed_out.output[:,-1,:], dim=-1, index=patch_answer_idxs.view(-1, 1)).squeeze(-1) + \
            t.gather(model.embed_out.output[:,-1,:], dim=-1, index=clean_answer_idxs.view(-1, 1)).squeeze(-1)
        )
    
    if args.ablation == 'resample': ablation_fn = lambda x: x
    if args.ablation == 'zero': ablation_fn = lambda x: x.zeros_like()
    if args.ablation == 'mean': ablation_fn = lambda x: x.mean(dim=0).expand_as(x)

    if args.faithfulness:
        ablation_outs = run_with_ablations(
            clean_inputs,
            patch_inputs,
            model,
            submodules,
            dictionaries,
            nodes,
            metric_fn,
            ablation_fn=ablation_fn,
            handle_resids=args.handle_resids
        )
        print(f"F(C) = {ablation_outs.mean()}")

        with model.trace(clean_inputs):
            metric = metric_fn(model).save()
        normal_outs = metric.value
        print(f"F(M) = {normal_outs.mean()}")

        all_ablated = run_with_ablations(
            clean_inputs,
            patch_inputs,
            model,
            submodules,
            dictionaries,
            nodes={submod : [] for submod in submodules},
            metric_fn=metric_fn,
            ablation_fn=ablation_fn,
            handle_resids=args.handle_resids
        )
        print(f"F(∅) = {all_ablated.mean()}")

        print(f"|F(C) - F(M)| = {(ablation_outs - normal_outs).abs().mean()}")
        print(f"|F(∅) - F(M)| = {(all_ablated - normal_outs).abs().mean()}")

        print(normal_outs - ablation_outs)


    if args.completeness:

        ablation_outs = run_with_ablations(
            clean_inputs,
            patch_inputs,
            model,
            submodules,
            dictionaries,
            nodes,
            metric_fn,
            complement=True,
            ablation_fn=ablation_fn,
            handle_resids=args.handle_resids
        )
        print(f"F(M \ C) = {ablation_outs.mean()}")
        print(f'|F(∅) - F(M \ C)| = {(all_ablated - ablation_outs).abs().mean()}')




        # print(f"Completeness: {(all_ablated - ablation_outs).abs().mean()}")



