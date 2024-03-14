import sys
sys.path.append('..')
import math
import os
import gc
import json
import torch as t
from tqdm import tqdm
from loading_utils import load_cluster_dict, load_examples_nopair
from circuit_triangles import get_circuit
from tqdm import trange
from nnsight import LanguageModel
from circuit_plotting import plot_circuit
from dictionary_learning import AutoEncoder
import argparse

def get_circuit_clusters(dataset,
                        model,
                        embed,
                        attns,
                        mlps,
                        resids,
                        dictionaries,
                        d_model=512,
                        dict_id=10,
                        dict_size=32768,
                        max_length=64,
                        max_examples=100,
                        batch_size=2,
                        node_threshold=0.1,
                        edge_threshold=0.01,
                        aggregation="sum",
                        device="cuda:0",
                        dataset_name="cluster_circuit",
                        circuit_dir="circuits/",
                        plot_dir="circuits/figures/"):

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
            aggregation=aggregation,
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

    # with open(f'circuits/{save_basename}_dict{dict_id}_node{node_threshold}_edge{edge_threshold}_n{num_examples}_aggsum.pt', 'rb') as infile:
    #     save_dict = t.load(infile)
    nodes = save_dict['nodes']
    edges = save_dict['edges']

    # feature annotations
    try:
        with open(f'{dict_id}_{dict_size}_annotations.json', 'r') as f:
            annotations = json.load(f)
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster_param_string", type=str, default="lin_effects_final-5-pos_nsamples8192_nctx64")
    parser.add_argument("--clusters_path", type=str, default="/home/can/feature_clustering/app_clusters/lin_effects_final-5-pos_nsamples8192_nctx64.json")
    parser.add_argument("--samples_path", type=str, default="/home/can/feature_clustering/clustering_pythia-70m-deduped_tloss0.1_nsamples8192_npos64_filtered-induction_attn-mlp-resid/samples8192.json")
    parser.add_argument("--n_total_clusters", type=int, default=750)
    parser.add_argument("--start_at_cluster", type=int, default=0)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--dict_size", type=int, default=32768)
    parser.add_argument("--dict_path", type=str, default="/share/projects/dictionary_circuits/autoencoders/pythia-70m-deduped/")
    parser.add_argument("--dict_id", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-70m-deduped")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--node_threshold", type=float, default=0.1)
    parser.add_argument("--edge_threshold", type=float, default=0.01)
    parser.add_argument("--aggregation", type=str, default="sum")
    args = parser.parse_args()

    # Load model and dictionaries
    model = LanguageModel(args.model_name, device_map=args.device, dispatch=True)

    embed = model.gpt_neox.embed_in
    attns = [layer.attention for layer in model.gpt_neox.layers]
    mlps = [layer.mlp for layer in model.gpt_neox.layers]
    resids = [layer for layer in model.gpt_neox.layers]
    # /om/user/ericjm/dictionary-circuits/pythia-70m-deduped/
    dictionaries = {}
    for i in range(len(model.gpt_neox.layers)):
        ae = AutoEncoder(args.d_model, args.dict_size).to(args.device)
        ae.load_state_dict(t.load(os.path.join(args.dict_path, f'embed/{args.dict_id}_{args.dict_size}/ae.pt')))
        dictionaries[embed] = ae

        ae = AutoEncoder(args.d_model, args.dict_size).to(args.device)
        ae.load_state_dict(t.load(os.path.join(args.dict_path, f'attn_out_layer{i}/{args.dict_id}_{args.dict_size}/ae.pt')))
        dictionaries[attns[i]] = ae

        ae = AutoEncoder(args.d_model, args.dict_size).to(args.device)
        ae.load_state_dict(t.load(os.path.join(args.dict_path, f'mlp_out_layer{i}/{args.dict_id}_{args.dict_size}/ae.pt')))
        dictionaries[mlps[i]] = ae

        ae = AutoEncoder(args.d_model, args.dict_size).to(args.device)
        ae.load_state_dict(t.load(os.path.join(args.dict_path, f'resid_out_layer{i}/{args.dict_id}_{args.dict_size}/ae.pt')))
        dictionaries[resids[i]] = ae

    for cluster_idx in trange(args.start_at_cluster, args.n_total_clusters):
        save_basename = "can_clusters_test"
        cluster_dataset = load_cluster_dict(
            samples_path=args.samples_path,
            clusters_map_path=args.clusters_path + args.cluster_param_string + ".json",
            n_total_clusters=args.n_total_clusters,
            cluster_idx=cluster_idx,
        )

        get_circuit_clusters(
            dataset=cluster_dataset,
            dataset_name=f"{args.cluster_param_string}_cluster{cluster_idx}of{args.n_total_clusters}",
            model=model,
            embed=embed,
            attns=attns,
            mlps=mlps,
            resids=resids,
            dictionaries=dictionaries,
            d_model=args.d_model,
            batch_size=args.batch_size,
            aggregation=args.aggregation,
            node_threshold=args.node_threshold,
            edge_threshold=args.edge_threshold,
            )