
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from loading_utils import load_examples
from attribution import patching_effect
from nnsight import LanguageModel
from dictionary_learning import AutoEncoder
import torch as t
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='EleutherAI/pythia-70m-deduped')
    parser.add_argument('--dict_path', type=str, default='dictionaries/pythia-70m-deduped/')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='rc_train')
    parser.add_argument('--num_examples', type=int, default=30)
    parser.add_argument('--dict_id', type=int, default=10)
    parser.add_argument('--activation_dim', type=int, default=512)
    parser.add_argument('--expansion_factor', type=int, default=64)
    parser.add_argument('--length', type=int, default=6)
    parser.add_argument('--save_dir', type=str, default='effects.pt')
    args = parser.parse_args()

    model = LanguageModel(args.model, device_map=args.device, dispatch=True)

    dataset = f'data/{args.dataset}.json'
    examples = load_examples(dataset, args.num_examples, model, length=args.length)
    clean_inputs = t.cat([e['clean_prefix'] for e in examples], dim=0).to(args.device)
    patch_inputs = t.cat([e['patch_prefix'] for e in examples], dim=0).to(args.device)
    clean_answer_idxs = t.tensor([e['clean_answer'] for e in examples], dtype=t.long, device=args.device)
    patch_answer_idxs = t.tensor([e['patch_answer'] for e in examples], dtype=t.long, device=args.device)

    out = {
        'examples' : examples
    }

    def metric_fn(model):
        return (
            t.gather(model.embed_out.output[:,-1,:], dim=-1, index=patch_answer_idxs.view(-1, 1)).squeeze(-1) - \
            t.gather(model.embed_out.output[:,-1,:], dim=-1, index=clean_answer_idxs.view(-1, 1)).squeeze(-1)
        )

    dictionary_size = args.activation_dim * args.expansion_factor

    submodules = []
    submod_names = {}
    dictionaries = {}
    submodules.append(model.gpt_neox.embed_in)
    submod_names[model.gpt_neox.embed_in] = 'embed'
    ae = AutoEncoder(args.activation_dim, dictionary_size).to(args.device)
    ae.load_state_dict(t.load(f'{args.dict_path}/embed/{args.dict_id}_{dictionary_size}/ae.pt'))
    dictionaries[model.gpt_neox.embed_in] = ae
    for i in range(len(model.gpt_neox.layers)):
        submodule = model.gpt_neox.layers[i].attention
        ae = AutoEncoder(args.activation_dim, dictionary_size).to(args.device)
        ae.load_state_dict(t.load(f'{args.dict_path}/attn_out_layer{i}/{args.dict_id}_{dictionary_size}/ae.pt'))
        submodules.append(submodule)
        submod_names[submodule] = f'attn_{i}'
        dictionaries[submodule] = ae

        submodule = model.gpt_neox.layers[i].mlp
        ae = AutoEncoder(args.activation_dim, dictionary_size).to(args.device)
        ae.load_state_dict(t.load(f'{args.dict_path}/mlp_out_layer{i}/{args.dict_id}_{dictionary_size}/ae.pt'))
        submodules.append(submodule)
        submod_names[submodule] = f'mlp_{i}'
        dictionaries[submodule] = ae

        submodule = model.gpt_neox.layers[i]
        ae = AutoEncoder(args.activation_dim, dictionary_size).to(args.device)
        ae.load_state_dict(t.load(f'{args.dict_path}/resid_out_layer{i}/{args.dict_id}_{dictionary_size}/ae.pt'))
        submodules.append(submodule)
        submod_names[submodule] = f'resid_{i}'
        dictionaries[submodule] = ae



    atp_effects = patching_effect(
        clean_inputs,
        patch_inputs,
        model,
        submodules,
        dictionaries,
        metric_fn,
        method='attrib'
    )

    ig_effects = patching_effect(
        clean_inputs,
        patch_inputs,
        model,
        submodules,
        dictionaries,
        metric_fn,
        method='ig'
    )

    exact_effects = patching_effect(
        clean_inputs,
        patch_inputs,
        model,
        submodules,
        dictionaries,
        metric_fn,
        method='exact'
    )

    out['atp_effects'] = {submod_names[k] : v for k, v in atp_effects.effects.items()}
    out['ig_effects'] = {submod_names[k] : v for k, v in ig_effects.effects.items()}
    out['exact_effects'] = {submod_names[k] : v for k, v in exact_effects.effects.items()}

    t.save(out, args.save_dir)

