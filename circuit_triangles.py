from dictionary_learning import AutoEncoder
from attribution import EffectOut
import torch as t
from nnsight import LanguageModel
from attribution_2 import patching_effect, jvp
from einops import rearrange
from activation_utils import SparseAct
from collections import defaultdict
import argparse
from circuit_plotting import plot_circuit
import json
import pickle
import os
from tkdict import TKDict
from tqdm import tqdm
from loading_utils import load_examples, load_examples_nopair

def flatten_index(idxs, shape):
    """
    index : a tensor of shape [n, len(shape)]
    shape : a shape
    return a tensor of shape [n] where each element is the flattened index
    """
    idxs = idxs.t()
    # get strides from shape
    strides = [1]
    for i in range(len(shape)-1, 0, -1):
        strides.append(strides[-1]*shape[i])
    strides = list(reversed(strides))
    strides = t.tensor(strides).to(idxs.device)
    # flatten index
    return (idxs * strides).sum(dim=1).unsqueeze(0)

def prod(l):
    out = 1
    for x in l: out *= x
    return out

def sparse_flatten(x):
    x = x.coalesce()
    return t.sparse_coo_tensor(
        flatten_index(x.indices(), x.shape),
        x.values(),
        (prod(x.shape),)
    )

def reshape_index(index, shape):
    """
    index : a tensor of shape [n]
    shape : a shape
    return a tensor of shape [n, len(shape)] where each element is the reshaped index
    """
    multi_index = []
    for dim in reversed(shape):
        multi_index.append(index % dim)
        index //= dim
    multi_index.reverse()
    return t.stack(multi_index, dim=-1)

def sparse_reshape(x, shape):
    """
    x : a sparse COO tensor
    shape : a shape
    return x reshaped to shape
    """
    # first flatten x
    x = sparse_flatten(x).coalesce()
    new_indices = reshape_index(x.indices()[0], shape)
    return t.sparse_coo_tensor(new_indices.t(), x.values(), shape)

def sparse_mean(x, dim):
    if isinstance(dim, int):
        return x.sum(dim=dim) / x.shape[dim]
    else:
        return x.sum(dim=dim) / prod(x.shape[d] for d in dim)

def get_circuit(
        clean,
        patch,
        model,
        attns,
        mlps,
        resids,
        dictionaries,
        metric_fn,
        metric_kwargs=dict(),
        aggregation='sum', # or 'none' for not aggregating across sequence position
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
        metric_fn,
        metric_kwargs=metric_kwargs,
        method='ig' # get better approximations for early layers by using ig
    )

    def unflatten(tensor): # will break if dictionaries vary in size between layers
        b, s, f = effects[resids[0]].act.shape
        unflattened = rearrange(tensor, '(b s x) -> b s x', b=b, s=s)
        return SparseAct(act=unflattened[...,:f], res=unflattened[...,f:])
    
    features_by_submod = {
        submod : (effects[submod].to_tensor().flatten().abs() > node_threshold).nonzero().flatten().tolist() for submod in all_submods
    }

    n_layers = len(resids)

    nodes = {'y' : total_effect}
    for i in range(n_layers):
        nodes[f'attn_{i}'] = effects[attns[i]]
        nodes[f'mlp_{i}'] = effects[mlps[i]]
        nodes[f'resid_{i}'] = effects[resids[i]]

    edges = defaultdict(lambda:{})
    edges[f'resid_{len(resids) - 1}'] = { 'y' : effects[resids[-1]].to_tensor().flatten().to_sparse() }

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

        edges[f'mlp_{layer}'][f'resid_{layer}'] = MR_effect
        edges[f'attn_{layer}'][f'resid_{layer}'] = AR_effect

        if layer > 0:
            prev_resid = resids[layer - 1]

            RM_effect, _ = N(prev_resid, mlp)
            RA_effect, _ = N(prev_resid, attn)

            edges[f'resid_{layer - 1}'][f'mlp_{layer}'] = RM_effect
            edges[f'resid_{layer - 1}'][f'attn_{layer}'] = RA_effect

            MR_grad = MR_grad.coalesce()
            AR_grad = AR_grad.coalesce()

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
            edges[f'resid_{layer - 1}'][f'resid_{layer}'] = RR_effect - RMR_effect - RAR_effect

    # rearrange weight matrices
    for child in edges:
        # get shape for child
        bc, sc, fc = nodes[child].act.shape
        for parent in edges[child]:
            weight_matrix = edges[child][parent]
            if parent == 'y':
                weight_matrix = sparse_reshape(weight_matrix, (bc, sc, fc+1))
            else:
                bp, sp, fp = nodes[parent].act.shape
                assert bp == bc
                weight_matrix = sparse_reshape(weight_matrix, (bp, sp, fp+1, bc, sc, fc+1))
            edges[child][parent] = weight_matrix
    
    if aggregation == 'sum':
        # aggregate across sequence position
        for child in edges:
            for parent in edges[child]:
                weight_matrix = edges[child][parent]
                if parent == 'y':
                    weight_matrix = weight_matrix.sum(dim=1)
                else:
                    weight_matrix = weight_matrix.sum(dim=(1, 4))
                edges[child][parent] = weight_matrix
        for node in nodes:
            if node != 'y':
                nodes[node] = nodes[node].sum(dim=1)

        # aggregate across batch dimension
        for child in edges:
            bc, fc = nodes[child].act.shape
            for parent in edges[child]:
                weight_matrix = edges[child][parent]
                if parent == 'y':
                    weight_matrix = weight_matrix.sum(dim=0) / bc
                else:
                    bp, fp = nodes[parent].act.shape
                    assert bp == bc
                    weight_matrix = weight_matrix.sum(dim=(0,2)) / bc
                edges[child][parent] = weight_matrix
        for node in nodes:
            if node != 'y':
                nodes[node] = nodes[node].mean(dim=0)
    
    elif aggregation == 'none':

        # aggregate across batch dimensions
        for child in edges:
            # get shape for child
            bc, sc, fc = nodes[child].act.shape
            for parent in edges[child]:
                weight_matrix = edges[child][parent]
                if parent == 'y':
                    weight_matrix = sparse_reshape(weight_matrix, (bc, sc, fc+1))
                    weight_matrix = weight_matrix.sum(dim=0) / bc
                else:
                    bp, sp, fp = nodes[parent].act.shape
                    assert bp == bc
                    weight_matrix = sparse_reshape(weight_matrix, (bp, sp, fp+1, bc, sc, fc+1))
                    weight_matrix = weight_matrix.sum(dim=(0, 3)) / bc
                edges[child][parent] = weight_matrix
        for node in nodes:
            nodes[node] = nodes[node].mean(dim=0)

    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")



    return nodes, edges

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tests', default=False, action='store_true')
    parser.add_argument('--dataset', '-d', type=str, default='simple')
    parser.add_argument('--num_examples', '-n', type=int, default=10)
    parser.add_argument('--example_length', '-l', type=int, default=None)
    parser.add_argument('--model', type=str, default='EleutherAI/pythia-70m-deduped')
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--dict_id', type=str, default=10)
    parser.add_argument('--dict_size', type=int, default=32768)
    parser.add_argument('--batches', type=int, default=1)
    parser.add_argument('--aggregation', type=str, default='sum')
    parser.add_argument('--node_threshold', type=float, default=0.1)
    parser.add_argument('--edge_threshold', type=float, default=0.01)
    parser.add_argument('--pen_thickness', type=float, default=1)
    parser.add_argument('--nopair', default=False, action="store_true")
    parser.add_argument('--plot_circuit', default=False, action='store_true')
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()


    device = args.device

    if args.tests:
        def _assert_equal(x, y):
            # assert that two sparse tensors are equal
            assert x.shape == y.shape
            assert t.all((x - y).coalesce().values() == 0)

        # test flatten_index
        shape = [4, 3, 1, 8, 2]
        x = t.randn(*shape).to(device)
        assert t.all(
            x.flatten().to_sparse().indices() == \
            flatten_index(x.to_sparse().indices(), shape)
        )

        # test sparse_flatten
        shape = [5]
        x = t.randn(*shape).to(device)
        _assert_equal(x.flatten().to_sparse(), sparse_flatten(x.to_sparse()))

        shape = [5, 2, 10, 3, 4]
        x = t.randn(*shape).to(device)
        _assert_equal(x.flatten().to_sparse(), sparse_flatten(x.to_sparse()))
        
        # test sparse_reshape
        shape = [2, 2, 3, 5, 4]
        x = t.randn(*shape).to(device)
        x_flat = x.flatten().to_sparse()
        x_reshaped = sparse_reshape(x_flat, shape)
        _assert_equal(x.to_sparse(), x_reshaped)

        shape = [4, 6]
        new_shape = [2, 2, 6]
        x = t.randn(*shape).to(device)
        x_rearranged = rearrange(x, '(a b) c -> a b c', a=2)
        assert t.all(x_rearranged.to_dense() == sparse_reshape(x.to_sparse(), new_shape).to_dense())

        shape = [4, 4, 4]
        new_shape = [2, 2, 4, 2, 2]
        x = t.randn(*shape).to(device)
        x_rearranged = rearrange(x, '(a b) c (d e) -> a b c d e', a=2, b=2, d=2, e=2)
        assert t.all(x_rearranged.to_dense() == sparse_reshape(x.to_sparse(), new_shape).to_dense())


    model = LanguageModel(args.model, device_map=device, dispatch=True)

    attns = [layer.attention for layer in model.gpt_neox.layers]
    mlps = [layer.mlp for layer in model.gpt_neox.layers]
    resids = [layer for layer in model.gpt_neox.layers]

    dictionaries = {}
    if args.dict_id == 'id':
        from dictionary_learning.dictionary import IdentityDict
        for i in range(len(model.gpt_neox.layers)):
            dictionaries[attns[i]] = IdentityDict(args.d_model)
            dictionaries[mlps[i]] = IdentityDict(args.d_model)
            dictionaries[resids[i]] = IdentityDict(args.d_model)
    else:
        for i in range(len(model.gpt_neox.layers)):
            ae = AutoEncoder(args.d_model, args.dict_size).to(device)
            ae.load_state_dict(t.load(f'/share/projects/dictionary_circuits/autoencoders/pythia-70m-deduped/attn_out_layer{i}/{args.dict_id}_{args.dict_size}/ae.pt'))
            dictionaries[attns[i]] = ae

            ae = AutoEncoder(args.d_model, args.dict_size).to(device)
            ae.load_state_dict(t.load(f'/share/projects/dictionary_circuits/autoencoders/pythia-70m-deduped/mlp_out_layer{i}/{args.dict_id}_{args.dict_size}/ae.pt'))
            dictionaries[mlps[i]] = ae

            ae = AutoEncoder(args.d_model, args.dict_size).to(device)
            ae.load_state_dict(t.load(f'/share/projects/dictionary_circuits/autoencoders/pythia-70m-deduped/resid_out_layer{i}/{args.dict_id}_{args.dict_size}/ae.pt'))
            dictionaries[resids[i]] = ae
    
    if args.nopair:
        data_path = f"contexts/{args.dataset}.json"
    else:
        data_path = f"/share/projects/dictionary_circuits/data/phenomena/{args.dataset}.json"

    if args.nopair:
        examples = load_examples_nopair(data_path, args.num_examples, model, length=args.example_length)
    else:
        examples = load_examples(data_path, args.num_examples, model, pad_to_length=args.example_length)

    batch_size = len(examples) // args.batches
    for batch in tqdm(range(args.batches), desc="Batches", total=args.batches):
        batch_examples = examples[batch*batch_size:(batch+1)*batch_size]
        clean_inputs = t.cat([e['clean_prefix'] for e in batch_examples], dim=0).to(device)
        clean_answer_idxs = t.tensor([e['clean_answer'] for e in batch_examples], dtype=t.long, device=device)

        if args.nopair:
            patch_inputs = None
            def metric_fn(model):
                return (
                    -1 * t.gather(model.embed_out.output[:,-1,:], dim=-1, index=clean_answer_idxs.view(-1, 1)).squeeze(-1)
                )
        else:
            patch_inputs = t.cat([e['patch_prefix'] for e in batch_examples], dim=0).to(device)
            patch_answer_idxs = t.tensor([e['patch_answer'] for e in batch_examples], dtype=t.long, device=device)
            def metric_fn(model):
                return (
                    t.gather(model.embed_out.output[:,-1,:], dim=-1, index=patch_answer_idxs.view(-1, 1)).squeeze(-1) - \
                    t.gather(model.embed_out.output[:,-1,:], dim=-1, index=clean_answer_idxs.view(-1, 1)).squeeze(-1)
                )
        
        nodes, edges = get_circuit(
            clean_inputs,
            patch_inputs,
            model,
            attns,
            mlps,
            resids,
            dictionaries,
            metric_fn,
            aggregation=args.aggregation,
            node_threshold=args.node_threshold,
            edge_threshold=args.edge_threshold,
        )

        save_file = f'circuits/{args.dataset}_dict{args.dict_id}_node{args.node_threshold}_edge{args.edge_threshold}_n{args.num_examples}_batch{batch}'

        with open(f"{save_file}.pt", "wb") as outfile:
            save_dict = {
                "examples" : batch_examples,
                "nodes": dict(nodes), 
                "edges": dict(edges)
            }
            t.save(save_dict, outfile)
        
        # memory cleanup
        del nodes, edges

    # aggregate over the batches
    out_dicts = [
        t.load(f'circuits/{args.dataset}_dict{args.dict_id}_node{args.node_threshold}_edge{args.edge_threshold}_n{args.num_examples}_batch{batch}.pt') for batch in range(args.batches)
    ]
    assert sum([len(d['examples']) for d in out_dicts]) == args.num_examples
    nodes = {
        k : sum([len(d['examples']) * d['nodes'][k] for d in out_dicts]) / args.num_examples for k in out_dicts[0]['nodes'].keys()
    }
    # need to do something funky to deal with the fact that edges are sparse tensors
    edges = {k : {v : len(out_dicts[0]['examples']) * out_dicts[0]['edges'][k][v] for v in out_dicts[0]['edges'][k].keys()} for k in out_dicts[0]['edges'].keys()}
    for k in out_dicts[0]['edges'].keys():
        for v in out_dicts[0]['edges'][k].keys():
            for d in out_dicts[1:]:
                edges[k][v] += len(d['examples']) * d['edges'][k][v]
            edges[k][v] = 1/args.num_examples * edges[k][v]

    save_dict = {
        "examples" : [e for d in out_dicts for e in d['examples']],
        "nodes": nodes,
        "edges": edges
    }
    with open(f'circuits/{args.dataset}_dict{args.dict_id}_node{args.node_threshold}_edge{args.edge_threshold}_n{args.num_examples}.pt', 'wb') as outfile:
        t.save(save_dict, outfile)

    # feature annotations
    try:
        with open(f'{args.dict_id}_{args.dict_size}_annotations.json', 'r') as f:
            annotations = json.load(f)
    except:
        annotations = None

    plot_circuit(
        nodes, 
        edges, 
        layers=len(model.gpt_neox.layers), 
        node_threshold=args.node_threshold, 
        edge_threshold=args.edge_threshold, 
        pen_thickness=args.pen_thickness, 
        annotations=annotations, 
        save_dir=f'circuits/figures/{args.dataset}_dict{args.dict_id}_node{args.node_threshold}_edge{args.edge_threshold}_n{args.num_examples}')
