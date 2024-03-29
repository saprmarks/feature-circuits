import argparse
import gc
import json
import math
import os
from collections import defaultdict

import torch as t
from einops import rearrange
from tqdm import tqdm

from activation_utils import SparseAct
from attribution import patching_effect, jvp
from circuit_plotting import plot_circuit
from dictionary_learning import AutoEncoder
from loading_utils import load_examples, load_examples_nopair
from nnsight import LanguageModel


###### utilities for dealing with sparse COO tensors ######
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

######## end sparse tensor utilities ########


def get_circuit(
        clean,
        patch,
        model,
        embed,
        attns,
        mlps,
        resids,
        dictionaries,
        metric_fn,
        metric_kwargs=dict(),
        aggregation='sum', # or 'none' for not aggregating across sequence position
        nodes_only=False,
        node_threshold=0.1,
        edge_threshold=0.01,
):
    all_submods = [embed] + [submod for layer_submods in zip(mlps, attns, resids) for submod in layer_submods]
    
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
    nodes['embed'] = effects[embed]
    for i in range(n_layers):
        nodes[f'attn_{i}'] = effects[attns[i]]
        nodes[f'mlp_{i}'] = effects[mlps[i]]
        nodes[f'resid_{i}'] = effects[resids[i]]

    if nodes_only:
        if aggregation == 'sum':
            for k in nodes:
                if k != 'y':
                    nodes[k] = nodes[k].sum(dim=1)
        nodes = {k : v.mean(dim=0) for k, v in nodes.items()}
        return nodes, None

    edges = defaultdict(lambda:{})
    edges[f'resid_{len(resids)-1}'] = { 'y' : effects[resids[-1]].to_tensor().flatten().to_sparse() }

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
            prev_resid = resids[layer-1]
        else:
            prev_resid = embed

        RM_effect, _ = N(prev_resid, mlp)
        RA_effect, _ = N(prev_resid, attn)

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

        if layer > 0: 
            edges[f'resid_{layer-1}'][f'mlp_{layer}'] = RM_effect
            edges[f'resid_{layer-1}'][f'attn_{layer}'] = RA_effect
            edges[f'resid_{layer-1}'][f'resid_{layer}'] = RR_effect - RMR_effect - RAR_effect
        else:
            edges['embed'][f'mlp_{layer}'] = RM_effect
            edges['embed'][f'attn_{layer}'] = RA_effect
            edges['embed'][f'resid_0'] = RR_effect - RMR_effect - RAR_effect

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


def get_circuit_cluster(dataset,
                        model_name="EleutherAI/pythia-70m-deduped",
                        d_model=512,
                        dict_id=10,
                        dict_size=32768,
                        max_length=64,
                        max_examples=100,
                        batch_size=2,
                        node_threshold=0.1,
                        edge_threshold=0.01,
                        device="cuda:0",
                        dict_path="dictionaries/pythia-70m-deduped/",
                        dataset_name="cluster_circuit",
                        circuit_dir="circuits/",
                        plot_dir="circuits/figures/",
                        model=None,
                        dictionaries=None,):
    
    model = LanguageModel(model_name, device_map=device, dispatch=True)

    embed = model.gpt_neox.embed_in
    attns = [layer.attention for layer in model.gpt_neox.layers]
    mlps = [layer.mlp for layer in model.gpt_neox.layers]
    resids = [layer for layer in model.gpt_neox.layers]
    dictionaries = {}
    for i in range(len(model.gpt_neox.layers)):
        ae = AutoEncoder(d_model, dict_size).to(device)
        ae.load_state_dict(t.load(os.path.join(dict_path, f'embed/{dict_id}_{dict_size}/ae.pt')))
        dictionaries[embed] = ae

        ae = AutoEncoder(d_model, dict_size).to(device)
        ae.load_state_dict(t.load(os.path.join(dict_path, f'attn_out_layer{i}/{dict_id}_{dict_size}/ae.pt')))
        dictionaries[attns[i]] = ae

        ae = AutoEncoder(d_model, dict_size).to(device)
        ae.load_state_dict(t.load(os.path.join(dict_path, f'mlp_out_layer{i}/{dict_id}_{dict_size}/ae.pt')))
        dictionaries[mlps[i]] = ae

        ae = AutoEncoder(d_model, dict_size).to(device)
        ae.load_state_dict(t.load(os.path.join(dict_path, f'resid_out_layer{i}/{dict_id}_{dict_size}/ae.pt')))
        dictionaries[resids[i]] = ae

    examples = load_examples_nopair(dataset, max_examples, model, length=max_length)

    num_examples = min(len(examples), max_examples)
    n_batches = math.ceil(num_examples / batch_size)
    batches = [
        examples[batch*batch_size:(batch+1)*batch_size] for batch in range(n_batches)
    ]
    if num_examples < max_examples: # warn the user
        print(f"Total number of examples is less than {max_examples}. Using {num_examples} examples instead.")

    running_nodes = None
    running_edges = None

    for batch in tqdm(batches, desc="Batches"):
        clean_inputs = t.cat([e['clean_prefix'] for e in batch], dim=0).to(device)
        clean_answer_idxs = t.tensor([e['clean_answer'] for e in batch], dtype=t.long, device=device)

        patch_inputs = None
        def metric_fn(model):
            return (
                -1 * t.gather(
                    t.nn.functional.log_softmax(model.embed_out.output[:,-1,:], dim=-1), dim=-1, index=clean_answer_idxs.view(-1, 1)
                ).squeeze(-1)
            )
        
        nodes, edges = get_circuit(
            clean_inputs,
            patch_inputs,
            model,
            embed,
            attns,
            mlps,
            resids,
            dictionaries,
            metric_fn,
            aggregation="sum",
            node_threshold=node_threshold,
            edge_threshold=edge_threshold,
        )

        if running_nodes is None:
            running_nodes = {k : len(batch) * nodes[k].to('cpu') for k in nodes.keys() if k != 'y'}
            running_edges = { k : { kk : len(batch) * edges[k][kk].to('cpu') for kk in edges[k].keys() } for k in edges.keys()}
        else:
            for k in nodes.keys():
                if k != 'y':
                    running_nodes[k] += len(batch) * nodes[k].to('cpu')
            for k in edges.keys():
                for v in edges[k].keys():
                    running_edges[k][v] += len(batch) * edges[k][v].to('cpu')
        
        # memory cleanup
        del nodes, edges
        gc.collect()

    nodes = {k : v.to(device) / num_examples for k, v in running_nodes.items()}
    edges = {k : {kk : 1/num_examples * v.to(device) for kk, v in running_edges[k].items()} for k in running_edges.keys()}

    save_dict = {
        "examples" : examples,
        "nodes": nodes,
        "edges": edges
    }
    save_basename = f"{dataset_name}_dict{dict_id}_node{node_threshold}_edge{edge_threshold}_n{num_examples}_aggsum"
    with open(f'{circuit_dir}/{save_basename}.pt', 'wb') as outfile:
        t.save(save_dict, outfile)

    nodes = save_dict['nodes']
    edges = save_dict['edges']

    # feature annotations
    try:
        annotations = {}
        with open(f'annotations/{dict_id}_{dict_size}.jsonl', 'r') as f:
            for line in f:
                line = json.loads(line)
                annotations[line['Name']] = line['Annotation']
    except:
        annotations = None

    plot_circuit(
        nodes, 
        edges, 
        layers=len(model.gpt_neox.layers), 
        node_threshold=node_threshold, 
        edge_threshold=edge_threshold, 
        pen_thickness=1, 
        annotations=annotations, 
        save_dir=os.path.join(plot_dir, save_basename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tests', default=False, action='store_true')
    parser.add_argument('--dataset', '-d', type=str, default='simple')
    parser.add_argument('--num_examples', '-n', type=int, default=10)
    parser.add_argument('--example_length', '-l', type=int, default=None)
    parser.add_argument('--model', type=str, default='EleutherAI/pythia-70m-deduped')
    parser.add_argument("--dict_path", type=str, default="dictionaries/pythia-70m-deduped/")
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--dict_id', type=str, default=10)
    parser.add_argument('--dict_size', type=int, default=32768)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--aggregation', type=str, default='sum')
    parser.add_argument('--node_threshold', type=float, default=0.2)
    parser.add_argument('--edge_threshold', type=float, default=0.02)
    parser.add_argument('--pen_thickness', type=float, default=1)
    parser.add_argument('--nopair', default=False, action="store_true")
    parser.add_argument('--plot_circuit', default=False, action='store_true')
    parser.add_argument('--nodes_only', default=False, action='store_true')
    parser.add_argument('--plot_only', action="store_true")
    parser.add_argument("--circuit_dir", type=str, default="circuits/")
    parser.add_argument("--plot_dir", type=str, default="circuits/figures/")
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

    embed = model.gpt_neox.embed_in
    attns = [layer.attention for layer in model.gpt_neox.layers]
    mlps = [layer.mlp for layer in model.gpt_neox.layers]
    resids = [layer for layer in model.gpt_neox.layers]

    dictionaries = {}
    if args.dict_id == 'id':
        from dictionary_learning.dictionary import IdentityDict
        dictionaries[embed] = IdentityDict(args.d_model)
        for i in range(len(model.gpt_neox.layers)):
            dictionaries[attns[i]] = IdentityDict(args.d_model)
            dictionaries[mlps[i]] = IdentityDict(args.d_model)
            dictionaries[resids[i]] = IdentityDict(args.d_model)
    else:
        ae = AutoEncoder(args.d_model, args.dict_size).to(device)
        ae.load_state_dict(t.load(f'{args.dict_path}/embed/{args.dict_id}_{args.dict_size}/ae.pt'))
        dictionaries[embed] = ae
        for i in range(len(model.gpt_neox.layers)):
            ae = AutoEncoder(args.d_model, args.dict_size).to(device)
            ae.load_state_dict(t.load(f'{args.dict_path}/attn_out_layer{i}/{args.dict_id}_{args.dict_size}/ae.pt'))
            dictionaries[attns[i]] = ae

            ae = AutoEncoder(args.d_model, args.dict_size).to(device)
            ae.load_state_dict(t.load(f'{args.dict_path}/mlp_out_layer{i}/{args.dict_id}_{args.dict_size}/ae.pt'))
            dictionaries[mlps[i]] = ae

            ae = AutoEncoder(args.d_model, args.dict_size).to(device)
            ae.load_state_dict(t.load(f'{args.dict_path}/resid_out_layer{i}/{args.dict_id}_{args.dict_size}/ae.pt'))
            dictionaries[resids[i]] = ae
    
    if args.nopair:
        data_path = args.dataset
        save_basename = os.path.splitext(os.path.basename(args.dataset))[0]
        examples = load_examples_nopair(data_path, args.num_examples, model, length=args.example_length)
    else:
        data_path = f"data/{args.dataset}.json"
        save_basename = args.dataset
        if args.aggregation == "sum":
            examples = load_examples(data_path, args.num_examples, model, pad_to_length=args.example_length)
        else:
            examples = load_examples(data_path, args.num_examples, model, length=args.example_length)
    
    batch_size = args.batch_size
    num_examples = min([args.num_examples, len(examples)])
    n_batches = math.ceil(num_examples / batch_size)
    batches = [
        examples[batch*batch_size:(batch+1)*batch_size] for batch in range(n_batches)
    ]
    if num_examples < args.num_examples: # warn the user
        print(f"Total number of examples is less than {args.num_examples}. Using {num_examples} examples instead.")

    if not args.plot_only:
        running_nodes = None
        running_edges = None

        for batch in tqdm(batches, desc="Batches"):
                
            clean_inputs = t.cat([e['clean_prefix'] for e in batch], dim=0).to(device)
            clean_answer_idxs = t.tensor([e['clean_answer'] for e in batch], dtype=t.long, device=device)

            if args.nopair:
                patch_inputs = None
                def metric_fn(model):
                    return (
                        -1 * t.gather(
                            t.nn.functional.log_softmax(model.embed_out.output[:,-1,:], dim=-1), dim=-1, index=clean_answer_idxs.view(-1, 1)
                        ).squeeze(-1)
                    )
            else:
                patch_inputs = t.cat([e['patch_prefix'] for e in batch], dim=0).to(device)
                patch_answer_idxs = t.tensor([e['patch_answer'] for e in batch], dtype=t.long, device=device)
                def metric_fn(model):
                    return (
                        t.gather(model.embed_out.output[:,-1,:], dim=-1, index=patch_answer_idxs.view(-1, 1)).squeeze(-1) - \
                        t.gather(model.embed_out.output[:,-1,:], dim=-1, index=clean_answer_idxs.view(-1, 1)).squeeze(-1)
                    )
            
            nodes, edges = get_circuit(
                clean_inputs,
                patch_inputs,
                model,
                embed,
                attns,
                mlps,
                resids,
                dictionaries,
                metric_fn,
                nodes_only=args.nodes_only,
                aggregation=args.aggregation,
                node_threshold=args.node_threshold,
                edge_threshold=args.edge_threshold,
            )

            if running_nodes is None:
                running_nodes = {k : len(batch) * nodes[k].to('cpu') for k in nodes.keys() if k != 'y'}
                if not args.nodes_only: running_edges = { k : { kk : len(batch) * edges[k][kk].to('cpu') for kk in edges[k].keys() } for k in edges.keys()}
            else:
                for k in nodes.keys():
                    if k != 'y':
                        running_nodes[k] += len(batch) * nodes[k].to('cpu')
                if not args.nodes_only:
                    for k in edges.keys():
                        for v in edges[k].keys():
                            running_edges[k][v] += len(batch) * edges[k][v].to('cpu')
            
            # memory cleanup
            del nodes, edges
            gc.collect()

        nodes = {k : v.to(device) / num_examples for k, v in running_nodes.items()}
        if not args.nodes_only: 
            edges = {k : {kk : 1/num_examples * v.to(device) for kk, v in running_edges[k].items()} for k in running_edges.keys()}
        else: edges = None

        save_dict = {
            "examples" : examples,
            "nodes": nodes,
            "edges": edges
        }
        with open(f'{args.circuit_dir}/{save_basename}_dict{args.dict_id}_node{args.node_threshold}_edge{args.edge_threshold}_n{num_examples}_agg{args.aggregation}.pt', 'wb') as outfile:
            t.save(save_dict, outfile)

    else:
        with open(f'{args.circuit_dir}/{save_basename}_dict{args.dict_id}_node{args.node_threshold}_edge{args.edge_threshold}_n{num_examples}_agg{args.aggregation}.pt', 'rb') as infile:
            save_dict = t.load(infile)
        nodes = save_dict['nodes']
        edges = save_dict['edges']

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
        save_dir=f'{args.plot_dir}/{save_basename}_dict{args.dict_id}_node{args.node_threshold}_edge{args.edge_threshold}_n{num_examples}_agg{args.aggregation}')