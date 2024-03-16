from nnsight import LanguageModel
import torch as t
from dictionary_learning import AutoEncoder
from dictionary_learning.dictionary import IdentityDict
from argparse import ArgumentParser
from activation_utils import SparseAct
from loading_utils import load_examples

# TODO make work if there are also sequence positions
# for now, assumes that the indices of nodes are integers, rather than, e.g. tuples
def run_with_ablations(
        clean,
        patch, # none if zero ablations
        model, 
        submodules, # list of submodules 
        dictionaries, # dictionaries[submodule] is an autoencoder for submodule's output
        nodes, # nodes[submodule] is a boolean SparseAct with True for the nodes to keep (or ablate if complement is True)
        metric_fn, # metric_fn(model, **metric_kwargs) -> t.Tensor
        metric_kwargs=dict(),
        complement=False, # if True, then use the complement of nodes
        ablation_fn=lambda x: x, # what to do to the patch hidden states to produce values for ablation, default resample
        handle_resids='default', # or 'remove' to zero ablate all; 'keep' to keep all
    ):

    if patch is None: patch = clean
    patch_states = {}
    with model.trace(patch), t.no_grad():
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            x = submodule.output
            if type(x.shape) == tuple:
                x = x[0]
            x_hat, f = dictionary(x, output_features=True)
            patch_states[submodule] = SparseAct(act=f, res=x - x_hat).save()
    patch_states = {k : ablation_fn(v.value) for k, v in patch_states.items()}


    with model.trace(clean), t.no_grad():
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            submod_nodes = nodes[submodule].clone()
            x = submodule.output
            is_tuple = type(x.shape) == tuple
            if is_tuple:
                x = x[0]
            f = dictionary.encode(x)
            res = x - dictionary(x)

            # ablate features
            if complement: submod_nodes = ~submod_nodes
            submod_nodes.resc = submod_nodes.resc.expand(*submod_nodes.resc.shape[:-1], res.shape[-1])
            if handle_resids == 'remove':
                submod_nodes.resc = t.zeros_like(submod_nodes.resc).to(t.bool)
            if handle_resids == 'keep':
                submod_nodes.resc = t.ones_like(submod_nodes.resc).to(t.bool)

            f[...,~submod_nodes.act] = patch_states[submodule].act[...,~submod_nodes.act]
            res[...,~submod_nodes.resc] = patch_states[submodule].res[...,~submod_nodes.resc]
            
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
    parser.add_argument('--data', type=str, default='/share/projects/dictionary_circuits/data/phenomena/rc_test.json')
    parser.add_argument('--num_examples', '-n', type=int, default=10)
    parser.add_argument('--length', '-l', type=int, default=6)
    parser.add_argument('--handle_resids', type=str, default='default')
    parser.add_argument('--start_layer', type=int, default=-1)
    parser.add_argument('--dict_id', default=10)
    args = parser.parse_args()

    model = LanguageModel('EleutherAI/pythia-70m-deduped', device_map='cuda:0', dispatch=True)

    submodules = []
    if args.start_layer < 0: submodules.append(model.gpt_neox.embed_in)
    for i in range(args.start_layer, len(model.gpt_neox.layers)):
        submodules.extend([
            model.gpt_neox.layers[i].attention,
            model.gpt_neox.layers[i].mlp,
            model.gpt_neox.layers[i]
        ])

    submod_names = {
        model.gpt_neox.embed_in : 'embed'
    }
    for i in range(len(model.gpt_neox.layers)):
        submod_names[model.gpt_neox.layers[i].attention] = f'attn_{i}'
        submod_names[model.gpt_neox.layers[i].mlp] = f'mlp_{i}'
        submod_names[model.gpt_neox.layers[i]] = f'resid_{i}'
    
    if args.dict_id == 10:
        dict_size = 32768
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

    elif args.dict_id == 'id':
        dict_size = 512
        dictionaries = {submod : IdentityDict(512).to('cuda:0') for submod in submodules}
    
    nodes = t.load(f'circuits/{args.circuit}')['nodes']
    nodes = {
        submod : nodes[submod_names[submod]].abs() > args.threshold for submod in submodules
    }
    n_features = sum([n.act.sum().item() for n in nodes.values()])
    n_errs = sum([n.resc.sum().item() for n in nodes.values()])
    print(f"# features = {n_features}")
    print(f"# triangles = {n_errs}")

    examples = load_examples(args.data, args.num_examples, model, length=args.length)
    clean_inputs = t.cat([e['clean_prefix'] for e in examples], dim=0).to('cuda:0')
    clean_answer_idxs = t.tensor([e['clean_answer'] for e in examples], dtype=t.long, device='cuda:0')
    patch_inputs = t.cat([e['patch_prefix'] for e in examples], dim=0).to('cuda:0')
    patch_answer_idxs = t.tensor([e['patch_answer'] for e in examples], dtype=t.long, device='cuda:0')
    def metric_fn(model):
        return (
            - t.gather(model.embed_out.output[:,-1,:], dim=-1, index=patch_answer_idxs.view(-1, 1)).squeeze(-1) + \
            t.gather(model.embed_out.output[:,-1,:], dim=-1, index=clean_answer_idxs.view(-1, 1)).squeeze(-1)
        )
    
    # if args.ablation == 'resample': ablation_fn = lambda x: x[t.randperm(x.act.shape[0])] # TODO this is wrong for SparseActs
    if args.ablation == 'zero': ablation_fn = lambda x: x.zeros_like()
    if args.ablation == 'mean': ablation_fn = lambda x: x.mean(dim=0).expand_as(x)

    with model.trace(clean_inputs):
        metric = metric_fn(model).save()
    fm = metric.value
    print(f"F(M) = {fm.mean()}")

    fc = run_with_ablations(
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
    print(f"F(C) = {fc.mean()}")

    fccomp = run_with_ablations(
        clean_inputs,
        patch_inputs,
        model,
        submodules,
        dictionaries,
        nodes,
        metric_fn,
        ablation_fn=ablation_fn,
        complement=True,
        handle_resids=args.handle_resids
    )
    print(f"F(C') = {fccomp.mean()}")

    fempty = run_with_ablations(
        clean_inputs,
        patch_inputs,
        model,
        submodules,
        dictionaries,
        nodes = {submod : SparseAct(act=t.zeros(dict_size, dtype=t.bool), resc=t.zeros(1, dtype=t.bool)).to('cuda:0') for submod in submodules},
        metric_fn=metric_fn,
        ablation_fn=ablation_fn,
        handle_resids=args.handle_resids
    )
    print(f"F(∅) = {fempty.mean()}")

    print(f"faithfulness = {(fc.mean() - fempty.mean()) / (fm.mean() - fempty.mean())}")
    print(f"completeness = {(fccomp.mean() - fempty.mean()) / (fm.mean() - fempty.mean())}")



