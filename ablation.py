from nnsight import LanguageModel
import torch as t
import math
from dictionary_learning import AutoEncoder
from dictionary_learning.dictionary import IdentityDict
from argparse import ArgumentParser
from activation_utils import SparseAct
from loading_utils import load_examples
from attribution import Submodule
from dictionary_loading_utils import load_saes_and_submodules
from tqdm import tqdm

def run_with_ablations(
        clean, # clean inputs
        patch, # patch inputs for use in computing ablation values
        model, # a nnsight LanguageModel
        submodules, # list of submodules 
        dictionaries, # dictionaries[submodule] is an autoencoder for submodule's output
        nodes, # nodes[submodule] is a boolean SparseAct with True for the nodes to keep (or ablate if complement is True)
        metric_fn, # metric_fn(model, **metric_kwargs) -> t.Tensor
        metric_kwargs=dict(),
        complement=False, # if True, then use the complement of nodes
        ablation_fn=lambda x: x.mean(dim=0).expand_as(x), # what to do to the patch hidden states to produce values for ablation, default mean ablation
        handle_errors='default', # or 'remove' to zero ablate all; 'keep' to keep all
    ):

    if patch is None: patch = clean
    patch_states = {}
    with model.trace(patch), t.no_grad():
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            x = submodule.get_activation()
            f = dictionary.encode(x)
            x_hat = dictionary.decode(f)
            patch_states[submodule] = SparseAct(act=f, res=x - x_hat).save()
    patch_states = {k : ablation_fn(v.value) for k, v in patch_states.items()}


    with model.trace(clean), t.no_grad():
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            submod_nodes = nodes[submodule.submodule].clone()
            x = submodule.get_activation()
            f = dictionary.encode(x)
            res = x - dictionary(x)

            # ablate features
            if complement: submod_nodes = ~submod_nodes
            submod_nodes.resc = submod_nodes.resc.expand(*submod_nodes.resc.shape[:-1], res.shape[-1])
            if handle_errors == 'remove':
                submod_nodes.resc = t.zeros_like(submod_nodes.resc).to(t.bool)
            if handle_errors == 'keep':
                submod_nodes.resc = t.ones_like(submod_nodes.resc).to(t.bool)

            f[...,~submod_nodes.act] = patch_states[submodule].act[...,~submod_nodes.act]
            res[...,~submod_nodes.resc] = patch_states[submodule].res[...,~submod_nodes.resc]
            
            submodule.set_activation(dictionary.decode(f) + res)
            # if is_tuple:
            #     submodule.output[0][:] = dictionary.decode(f) + res
            # else:
            #     submodule.output = dictionary.decode(f) + res

        metric = metric_fn(model, **metric_kwargs).save()
    return metric.value


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default="EleutherAI/pythia-70m-deduped",
                        help="Name of model on which we evaluate faithfulness.")
    parser.add_argument('--threshold', type=float, default=0.1,
                        help="Node threshold for the circuit.")
    parser.add_argument('--ablation', type=str, default='mean',
                        help="Ablation style. Can be one of `mean`, `resample`, `zero`.")
    parser.add_argument('--circuit', type=str,
                        help="Path to a circuit .pt file.")
    parser.add_argument('--data', type=str, default='rc_test.json',
                        help="Data on which to evaluate the circuit.")
    parser.add_argument('--batch_size', type=int, default=8,
                        help="Batch size for processing examples.")
    parser.add_argument('--num_examples', '-n', type=int, default=100,
                        help="Number of examples over which to evaluate the circuit.")
    parser.add_argument('--length', '-l', type=int, default=6,
                        help="Length of evaluation examples.")
    parser.add_argument('--handle_errors', type=str, default='default',
                        help="How to treat SAE error terms. Can be `default`, `keep`, or `remove`.")
    parser.add_argument('--start_layer', type=int, default=-1,
                        help="Layer to evaluate the circuit from. Layers below --start_layer are given to the model for free.")
    parser.add_argument('--dict_path', type=str, default='dictionaries/pythia-70m-deduped/',
                        help="Path to trained dictionaries.")
    parser.add_argument('--dict_id', default=10)
    parser.add_argument('--dict_size', type=int, default=32768,
                        help="Width of dictionary encoders.")
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    DEVICE = "cuda:0"

    if "gemma-2" in args.model:
        model = LanguageModel(args.model, device_map=args.device, dispatch=True,
                              torch_dtype=t.bfloat16, attn_implementation="eager")
        embed_submod = Submodule(
            name = "embed",
            submodule=model.model.embed_tokens,
        )
        model_layers = model.model.layers
        out_submod = model.lm_head
    else:
        model = LanguageModel(args.model, device_map=args.device, dispatch=True)
        embed_submod = Submodule(
            name="embed",
            submodule=model.gpt_neox.embed_in
        )
        model_layers = model.gpt_neox.layers
        out_submod = model.embed_out

    # submodules = []
    # if args.start_layer < 0: submodules.append(embed_submod)
    # for i in range(args.start_layer, len(model_layers)):
    #     if "gemma-2" in args.model:
    #         submodules.extend([
    #             Submodule(submodule=model.model.layers[i].self_attn.o_proj, use_input=True, name=f"attn_{i}"),
    #             Submodule(submodule=model.model.layers[i].post_feedforward_layernorm, name=f"mlp_{i}"),
    #             Submodule(submodule=model.model.layers[i], is_tuple=True, name=f"resid_{i}")
    #         ])
    #     else:
    #         submodules.extend([
    #             Submodule(submodule=model.gpt_neox.layers[i].attention, name=f"attn_{i}"),
    #             Submodule(model.gpt_neox.layers[i].mlp, name=f"mlp_{i}"),
    #             Submodule(model.gpt_neox.layers[i], name=f"resid_{i}", is_tuple=True)
    #         ])

    # submod_names = {
    #     embed_submod.submodule : 'embed'
    # }
    # for i in range(len(model.gpt_neox.layers)):
    #     submod_names[model.gpt_neox.layers[i].attention] = f'attn_{i}'
    #     submod_names[model.gpt_neox.layers[i].mlp] = f'mlp_{i}'
    #     submod_names[model.gpt_neox.layers[i]] = f'resid_{i}'
    
    if args.dict_id != 'id':
        if "gemma-2" in args.model:
            include_embed = False
        else:
            include_embed = True
        dict_size = args.dict_size
        submodules, dictionaries = load_saes_and_submodules(
            model, 
            args.model, 
            dtype=t.bfloat16,
            device=DEVICE,
            include_embed=include_embed
        )
        submodules = [s for s in submodules if int(s.name.split("_")[-1]) > args.start_layer]
        # dictionaries = {}
        # dictionaries[embed_submod] = AutoEncoder.from_pretrained(
        #     f'{args.dict_path}/embed/{args.dict_id}_{dict_size}/ae.pt',
        #     device=args.device
        # )
        # for i in range(len(model_layers)):
        #     dictionaries[model_layers[i].attention] = AutoEncoder.from_pretrained(
        #         f'{args.dict_path}/attn_out_layer{i}/{args.dict_id}_{dict_size}/ae.pt',
        #         device=args.device
        #     )
        #     dictionaries[model_layers[i].mlp] = AutoEncoder.from_pretrained(
        #         f'{args.dict_path}/mlp_out_layer{i}/{args.dict_id}_{dict_size}/ae.pt',
        #         device=args.device
        #     )
        #     dictionaries[model_layers[i]] = AutoEncoder.from_pretrained(
        #         f'{args.dict_path}/resid_out_layer{i}/{args.dict_id}_{dict_size}/ae.pt',
        #         device=args.device
        #     )

    elif args.dict_id == 'id':
        dict_size = 512 if "pythia-70m" in args.model else 2304     # TODO: probably won't work for Gemma attention
        dictionaries = {submod : IdentityDict(dict_size).to(args.device) for submod in submodules}
    
    nodes = t.load(args.circuit)['nodes']
    nodes = {
        submod.submodule : nodes[submod.name].abs() > args.threshold for submod in submodules
    }
    n_features = sum([n.act.sum().item() for n in nodes.values()])
    n_errs = sum([n.resc.sum().item() for n in nodes.values()])
    print(f"# features = {n_features}")
    print(f"# triangles = {n_errs}")

    examples = load_examples(f'data/{args.data}', args.num_examples, model, length=args.length)
    batch_size = args.batch_size
    num_examples = min([args.num_examples, len(examples)])
    n_batches = math.ceil(num_examples / batch_size)
    batches = [
        examples[batch*batch_size:(batch+1)*batch_size] for batch in range(n_batches)
    ]
    fm_list = None
    fc_list = None
    fccomp_list = None
    fempty_list = None

    for batch in tqdm(batches, desc="Batches"):   
        clean_inputs = t.cat([e['clean_prefix'] for e in batch], dim=0).to(args.device)
        clean_answer_idxs = t.tensor([e['clean_answer'] for e in batch], dtype=t.long, device=args.device)
        patch_inputs = t.cat([e['patch_prefix'] for e in batch], dim=0).to(args.device)
        patch_answer_idxs = t.tensor([e['patch_answer'] for e in batch], dtype=t.long, device=args.device)
        def metric_fn(model):
            return (
                - t.gather(out_submod.output[:,-1,:], dim=-1, index=patch_answer_idxs.view(-1, 1)).squeeze(-1) + \
                t.gather(out_submod.output[:,-1,:], dim=-1, index=clean_answer_idxs.view(-1, 1)).squeeze(-1)
            )
        
        if args.ablation == 'resample': 
            def ablation_fn(x):
                idxs = t.multinomial(t.ones(x.act.shape[0]), x.act.shape[0], replacement=True).to(x.act.device)
                return SparseAct(act=x.act[idxs], res=x.res[idxs])
        if args.ablation == 'zero': ablation_fn = lambda x: x.zeros_like()
        if args.ablation == 'mean': ablation_fn = lambda x: x.mean(dim=0).expand_as(x)

        with model.trace(clean_inputs):
            metric = metric_fn(model).save()
        fm = metric.value
        if fm_list is None:
            fm_list = fm
        else:
            fm_list = t.cat((fm_list, fm))
        # print(f"F(M) = {fm.mean()}")

        fc = run_with_ablations(
            clean_inputs,
            patch_inputs,
            model,
            submodules,
            dictionaries,
            nodes,
            metric_fn,
            ablation_fn=ablation_fn,
            handle_errors=args.handle_errors
        )
        if fc_list is None:
            fc_list = fc
        else:
            fc_list = t.cat((fc_list, fc))
        # print(f"F(C) = {fc.mean()}")

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
            handle_errors=args.handle_errors
        )
        if fccomp_list is None:
            fccomp_list = fccomp
        else:
            fccomp_list = t.cat((fccomp_list, fccomp))
        # print(f"F(C') = {fccomp.mean()}")

        fempty = run_with_ablations(
            clean_inputs,
            patch_inputs,
            model,
            submodules,
            dictionaries,
            nodes = {submod.submodule : SparseAct(act=t.zeros(dict_size, dtype=t.bool), resc=t.zeros(1, dtype=t.bool)).to(args.device) for submod in submodules},
            metric_fn=metric_fn,
            ablation_fn=ablation_fn,
            handle_errors=args.handle_errors
        )
        if fempty_list is None:
            fempty_list = fempty
        else:
            fempty_list = t.cat((fempty_list, fempty))
        # print(f"F(∅) = {fempty.mean()}")

    print(f"faithfulness = {(fc_list.mean() - fempty_list.mean()) / (fm_list.mean() - fempty_list.mean())}")
    print(f"completeness = {(fccomp_list.mean() - fempty_list.mean()) / (fm_list.mean() - fempty_list.mean())}")