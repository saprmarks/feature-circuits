import argparse
import gc
import json
import math
import os
from collections import defaultdict

import torch as t
from tqdm import tqdm

from attribution import patching_effect, jvp
from circuit_plotting import plot_circuit, plot_circuit_posaligned
from dictionary_learning import AutoEncoder
from data_loading_utils import load_examples, load_examples_nopair
from dictionary_loading_utils import load_saes_and_submodules
from nnsight import LanguageModel
from coo_utils import sparse_reshape

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
    aggregation="sum",  # or "none" for not aggregating across sequence position
    nodes_only=False,
    parallel_attn=False,
    node_threshold=0.1,
):
    all_submods = ([embed] if embed is not None else []) + [
        submod for layer_submods in zip(attns, mlps, resids) for submod in layer_submods
    ]

    # first get the patching effect of everything on y
    effects, deltas, grads, total_effect = patching_effect(
        clean,
        patch,
        model,
        all_submods,
        dictionaries,
        metric_fn,
        metric_kwargs=metric_kwargs,
        method="ig",  # get better approximations for early layers by using ig
    )

    features_by_submod = {
        submod: effects[submod].abs() > node_threshold for submod in all_submods
    }

    n_layers = len(resids)

    nodes = {"y": total_effect}
    if embed is not None:
        nodes["embed"] = effects[embed].detach()
    for i in range(n_layers):
        nodes[f"attn_{i}"] = effects[attns[i]].detach()
        nodes[f"mlp_{i}"] = effects[mlps[i]].detach()
        nodes[f"resid_{i}"] = effects[resids[i]].detach()

    del effects
    t.cuda.empty_cache()

    if nodes_only:
        if aggregation == "sum":
            for k in nodes:
                if k != "y":
                    nodes[k] = nodes[k].sum(dim=1)
        nodes = {k: v.mean(dim=0) for k, v in nodes.items()}
        del deltas, grads
        t.cuda.empty_cache()
        return nodes, None

    edges = defaultdict(lambda: {})
    edges[f"resid_{len(resids) - 1}"] = {
        "y": nodes[f"resid_{len(resids) - 1}"].to_tensor().flatten().to_sparse()
    }

    def N(upstream, downstream, midstream=[]):
        result = jvp(
            clean,
            model,
            dictionaries,
            downstream,
            features_by_submod[downstream],
            upstream,
            grads[downstream],
            deltas[upstream],
            intermediate_stopgrads=midstream,
        )
        return result.detach()  # Detach immediately to save memory

    # now we work backward through the model to get the edges
    for layer in reversed(range(len(resids))):
        resid = resids[layer]
        mlp = mlps[layer]
        attn = attns[layer]

        # Process each edge computation separately with cleanup
        MR_effect = N(mlp, resid)
        edges[f"mlp_{layer}"][f"resid_{layer}"] = MR_effect
        del MR_effect
        t.cuda.empty_cache()

        AR_effect = N(attn, resid, [mlp])
        edges[f"attn_{layer}"][f"resid_{layer}"] = AR_effect
        del AR_effect
        t.cuda.empty_cache()

        if not parallel_attn:
            AM_effect = N(attn, mlp)
            edges[f"attn_{layer}"][f"mlp_{layer}"] = AM_effect
            del AM_effect
            t.cuda.empty_cache()

        if layer > 0:
            prev_resid = resids[layer - 1]
        else:
            prev_resid = embed

        if prev_resid is not None:
            RM_effect = N(prev_resid, mlp, [attn])
            RA_effect = N(prev_resid, attn)
            RR_effect = N(prev_resid, resid, [mlp, attn])

            if layer > 0:
                edges[f"resid_{layer - 1}"][f"mlp_{layer}"] = RM_effect
                edges[f"resid_{layer - 1}"][f"attn_{layer}"] = RA_effect
                edges[f"resid_{layer - 1}"][f"resid_{layer}"] = RR_effect
            else:
                edges["embed"][f"mlp_{layer}"] = RM_effect
                edges["embed"][f"attn_{layer}"] = RA_effect
                edges["embed"]["resid_0"] = RR_effect

            del RM_effect, RA_effect, RR_effect
            t.cuda.empty_cache()

    del deltas, grads
    t.cuda.empty_cache()

    # rearrange weight matrices
    for child in edges:
        # get shape for child
        bc, sc, fc = nodes[child].act.shape
        for parent in edges[child]:
            weight_matrix = edges[child][parent]
            if parent == "y":
                weight_matrix = sparse_reshape(weight_matrix, (bc, sc, fc + 1))
            else:
                continue
            edges[child][parent] = weight_matrix

    if aggregation == "sum":
        # aggregate across sequence position
        for child in edges:
            for parent in edges[child]:
                weight_matrix = edges[child][parent]
                if parent == "y":
                    weight_matrix = weight_matrix.sum(dim=1)
                else:
                    weight_matrix = weight_matrix.sum(dim=(1, 4))
                edges[child][parent] = weight_matrix
        for node in nodes:
            if node != "y":
                nodes[node] = nodes[node].sum(dim=1)

        # aggregate across batch dimension
        for child in edges:
            bc, _ = nodes[child].act.shape
            for parent in edges[child]:
                weight_matrix = edges[child][parent]
                if parent == "y":
                    weight_matrix = weight_matrix.sum(dim=0) / bc
                else:
                    bp, _ = nodes[parent].act.shape
                    assert bp == bc
                    weight_matrix = weight_matrix.sum(dim=(0, 2)) / bc
                edges[child][parent] = weight_matrix
        for node in nodes:
            if node != "y":
                nodes[node] = nodes[node].mean(dim=0)

    elif aggregation == "none":
        # aggregate across batch dimensions
        for child in edges:
            # get shape for child
            bc, sc, fc = nodes[child].act.shape
            for parent in edges[child]:
                weight_matrix = edges[child][parent]
                if parent == "y":
                    weight_matrix = sparse_reshape(weight_matrix, (bc, sc, fc + 1))
                    weight_matrix = weight_matrix.sum(dim=0) / bc
                else:
                    bp, sp, fp = nodes[parent].act.shape
                    assert bp == bc
                    weight_matrix = weight_matrix.sum(dim=(0, 3)) / bc
                edges[child][parent] = weight_matrix
        for node in nodes:
            nodes[node] = nodes[node].mean(dim=0)

    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    return nodes, edges


def get_circuit_cluster(
    dataset,
    model_name="EleutherAI/pythia-70m-deduped",
    d_model=512,
    dict_id=10,
    dict_size=32768,
    max_length=100,
    max_examples=100,
    batch_size=1,
    node_threshold=0.1,
    edge_threshold=0.01,
    device="cuda:0",
    dict_path="dictionaries/pythia-70m-deduped/",
    dataset_name="cluster_circuit",
    circuit_dir="circuits/",
    plot_dir="circuits/figures/",
    model=None,
    dictionaries=None,
    create_plots=False,
):
    
    n_layers = {
        "EleutherAI/pythia-70m-deduped": 6,
        "google/gemma-2-2b": 26,
    }[model_name]
    parallel_attn = {
        "EleutherAI/pythia-70m-deduped": True,
        "google/gemma-2-2b": False,
    }[model_name]
    include_embed = {
        "EleutherAI/pythia-70m-deduped": True,
        "google/gemma-2-2b": False,
    }[model_name]
    dtype = {
        "EleutherAI/pythia-70m-deduped": t.float32,
        "google/gemma-2-2b": t.bfloat16,
    }[model_name]

    if model_name == "EleutherAI/pythia-70m-deduped":
        model = LanguageModel(model_name, device_map=device, dispatch=True, torch_dtype=dtype)
    elif model_name == "google/gemma-2-2b":
        model = LanguageModel(model_name, device_map=device, dispatch=True, attn_implementation="eager", torch_dtype=dtype)
    
    submodules, dictionaries = load_saes_and_submodules(
        model,
        separate_by_type=True,
        include_embed=include_embed,
        device=device,
        dtype=dtype,
    )

    examples = load_examples_nopair(dataset, max_examples, model)
    num_examples = min(len(examples), max_examples)
    n_batches = math.ceil(num_examples / batch_size)
    batches = [
        examples[batch * batch_size : (batch + 1) * batch_size]
        for batch in range(n_batches)
    ]
    if num_examples < max_examples:  # warn the user
        print(
            f"Total number of examples is less than {max_examples}. Using {num_examples} examples instead."
        )

    running_nodes = None
    running_edges = None

    for batch in tqdm(batches, desc="Batches"):
        clean_inputs = [e["clean_prefix"] for e in batch]
        clean_answer_idxs = t.tensor(
            [model.tokenizer(e["clean_answer"]).input_ids[-1] for e in batch],
            dtype=t.long,
            device=device
        )
        clean_inputs = model.tokenizer(
            clean_inputs, 
            return_tensors="pt", 
            padding=True,
            padding_side="left",
        ).input_ids
        clean_inputs = clean_inputs[:, -max_length:] #truncate on the left
        patch_inputs = None

        def metric_fn(model):
            return -1 * t.gather(
                model.output.logits[:, -1, :],
                dim=-1,
                index=clean_answer_idxs.view(-1, 1),
            ).squeeze(-1)

        nodes, edges = get_circuit(
            clean_inputs,
            patch_inputs,
            model,
            submodules.embed,
            submodules.attns,
            submodules.mlps,
            submodules.resids,
            dictionaries,
            metric_fn,
            aggregation="sum",
            node_threshold=node_threshold,
            # edge_threshold=edge_threshold,
            parallel_attn=parallel_attn,
        )

        if running_nodes is None:
            running_nodes = {
                k: len(batch) * nodes[k].to("cpu") for k in nodes.keys() if k != "y"
            }
            running_edges = {
                k: {kk: len(batch) * edges[k][kk].to("cpu") for kk in edges[k].keys()}
                for k in edges.keys()
            }
        else:
            for k in nodes.keys():
                if k != "y":
                    running_nodes[k] += len(batch) * nodes[k].to("cpu")
            for k in edges.keys():
                for v in edges[k].keys():
                    running_edges[k][v] += len(batch) * edges[k][v].to("cpu")

        # memory cleanup
        del nodes, edges
        gc.collect()

    nodes = {k: v.to(device) / num_examples for k, v in running_nodes.items()}
    edges = {
        k: {kk: 1 / num_examples * v.to(device) for kk, v in running_edges[k].items()}
        for k in running_edges.keys()
    }

    save_dict = {"examples": examples, "nodes": nodes, "edges": edges}
    save_basename = f"{dataset_name}_dict{dict_id}_node{node_threshold}_edge{edge_threshold}_n{num_examples}_aggsum"
    
    # Create the full path for saving the circuit file
    save_path = os.path.join(circuit_dir, save_basename + ".pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, "wb") as outfile:
        t.save(save_dict, outfile)


    if create_plots:
        nodes = save_dict["nodes"]
        edges = save_dict["edges"]
        annotations = None

        # Create the full path for saving the plot
        plot_save_path = os.path.join(plot_dir, save_basename)
        os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)

        plot_circuit(
            nodes,
            edges,
            layers=n_layers,
            node_threshold=node_threshold,
            edge_threshold=edge_threshold,
            pen_thickness=1,
            annotations=annotations,
            save_dir=plot_save_path,
            gemma_mode=(model_name == "google/gemma-2-2b"),
            parallel_attn=parallel_attn,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="simple_train",
        help="A subject-verb agreement dataset in data/, or a path to a cluster .json.",
    )
    parser.add_argument(
        "--num_examples",
        "-n",
        type=int,
        default=100,
        help="The number of examples from the --dataset over which to average indirect effects.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="EleutherAI/pythia-70m-deduped",
        help="The Huggingface ID of the model you wish to test.",
    )
    parser.add_argument(
        "--dict_path",
        type=str,
        default="dictionaries/pythia-70m-deduped/",
        help="Path to all dictionaries for your language model.",
    )
    parser.add_argument(
        "--d_model", type=int, default=512, help="Hidden size of the language model."
    )
    parser.add_argument(
        "--use_neurons",
        default=False,
        action="store_true",
        help="Use neurons instead of features.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Number of examples to process at once when running circuit discovery.",
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        default="sum",
        help="Aggregation across token positions. Should be one of `sum` or `none`.",
    )
    parser.add_argument(
        "--node_threshold",
        type=float,
        default=0.2,
        help="Indirect effect threshold for keeping circuit nodes.",
    )
    parser.add_argument(
        "--edge_threshold",
        type=float,
        default=0.02,
        help="Indirect effect threshold for keeping edges.",
    )
    parser.add_argument(
        "--pen_thickness",
        type=float,
        default=1,
        help="Scales the width of the edges in the circuit plot.",
    )
    parser.add_argument(
        "--nopair",
        default=False,
        action="store_true",
        help="Use if your data does not contain contrastive (minimal) pairs.",
    )
    parser.add_argument(
        "--plot_circuit",
        default=False,
        action="store_true",
        help="Plot the circuit after discovering it.",
    )
    parser.add_argument(
        "--nodes_only",
        default=False,
        action="store_true",
        help="Only search for causally implicated features; do not draw edges.",
    )
    parser.add_argument(
        "--plot_only",
        default=False,
        action="store_true",
        help="Do not run circuit discovery; just plot an existing circuit.",
    )
    parser.add_argument(
        "--circuit_dir",
        type=str,
        default="circuits",
        help="Directory to save/load circuits.",
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default="circuits/figures/",
        help="Directory to save figures.",
    )
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = t.device(args.device)

    n_layers = {
        "EleutherAI/pythia-70m-deduped": 6,
        "google/gemma-2-2b": 26,
    }[args.model]
    parallel_attn = {
        "EleutherAI/pythia-70m-deduped": True,
        "google/gemma-2-2b": False,
    }[args.model]
    include_embed = {
        "EleutherAI/pythia-70m-deduped": True,
        "google/gemma-2-2b": False,
    }[args.model]
    dtype = {
        "EleutherAI/pythia-70m-deduped": t.float32,
        "google/gemma-2-2b": t.bfloat16,
    }[args.model]

    if args.model == "EleutherAI/pythia-70m-deduped":
        model = LanguageModel(args.model, device_map=device, dispatch=True, torch_dtype=dtype)
    elif args.model == "google/gemma-2-2b":
        model = LanguageModel(
            args.model,
            device_map=device,
            dispatch=True,
            attn_implementation="eager",
            torch_dtype=dtype,
        )

    if args.nopair:
        data_path = f"data/{args.dataset}.json"
        examples = load_examples_nopair(
            data_path, args.num_examples, model
        )
    else:
        data_path = f"data/{args.dataset}.json"
        examples = load_examples(
            data_path, args.num_examples, model, use_min_length_only=True
        )

    num_examples = min([args.num_examples, len(examples)])
    if num_examples < args.num_examples:  # warn the user
        print(
            f"Total number of examples is less than {args.num_examples}. Using {num_examples} examples instead."
        )

    batch_size = args.batch_size
    n_batches = math.ceil(num_examples / batch_size)
    batches = [
        examples[batch * batch_size : (batch + 1) * batch_size]
        for batch in range(n_batches)
    ]

    loaded_from_disk = False
    save_base = (
        f"{args.model.split('/')[-1]}_{args.dataset}_n{num_examples}_agg{args.aggregation}"
        + ("_neurons" if args.use_neurons else "")
    )
    node_suffix = f"node{args.node_threshold}" if not args.nodes_only else "nodeall"
    if os.path.exists(save_path := f"{args.circuit_dir}/{save_base}_{node_suffix}.pt"):
        print(f"Loading circuit from {save_path}")
        with open(save_path, "rb") as infile:
            save_dict = t.load(infile, weights_only=False)
        nodes = save_dict["nodes"]
        edges = save_dict["edges"]
        loaded_from_disk = True
    elif not args.nodes_only:
        for f in os.listdir(args.circuit_dir):
            if "nodeall" in f:
                continue
            if f.startswith(save_base):
                node_thresh = float(f.split(".")[0].split("_node")[-1])
                if node_thresh < args.node_threshold:
                    print(f"Loading circuit from {args.circuit_dir}/{f}")
                    with open(f"{args.circuit_dir}/{f}", "rb") as infile:
                        save_dict = t.load(infile)
                    nodes = save_dict["nodes"]
                    edges = save_dict["edges"]
                    loaded_from_disk = True
                    break

    if not loaded_from_disk:
        print("computing circuit")
        submodules, dictionaries = load_saes_and_submodules(
            model,
            separate_by_type=True,
            include_embed=include_embed,
            neurons=args.use_neurons,
            device=device,
            dtype=dtype,
        )

        running_nodes = None
        running_edges = None

        for batch in tqdm(batches, desc="Batches"):
            clean_inputs = [e["clean_prefix"] for e in batch]
            clean_answer_idxs = t.tensor(
                [model.tokenizer(e["clean_answer"]).input_ids[-1] for e in batch],
                dtype=t.long,
                device=device,
            )

            if args.nopair:
                patch_inputs = None

                def metric_fn(model):
                    return -1 * t.gather(
                        model.output.logits[:, -1, :],
                        dim=-1,
                        index=clean_answer_idxs.view(-1, 1),
                    ).squeeze(-1)
            else:
                patch_inputs = [e["patch_prefix"] for e in batch]
                patch_answer_idxs = t.tensor(
                    [model.tokenizer(e["patch_answer"]).input_ids[-1] for e in batch],
                    dtype=t.long,
                    device=device,
                )

                def metric_fn(model):
                    logits = model.output.logits[:, -1, :]
                    return t.gather(
                        logits, dim=-1, index=patch_answer_idxs.view(-1, 1)
                    ).squeeze(-1) - t.gather(
                        logits, dim=-1, index=clean_answer_idxs.view(-1, 1)
                    ).squeeze(-1)

            nodes, edges = get_circuit(
                clean_inputs,
                patch_inputs,
                model,
                submodules.embed,
                submodules.attns,
                submodules.mlps,
                submodules.resids,
                dictionaries,
                metric_fn,
                nodes_only=args.nodes_only,
                aggregation=args.aggregation,
                node_threshold=args.node_threshold,
                parallel_attn=parallel_attn,
            )

            if running_nodes is None:
                running_nodes = {
                    k: len(batch) * nodes[k].to("cpu") for k in nodes.keys() if k != "y"
                }
                if not args.nodes_only:
                    running_edges = {
                        k: {
                            kk: len(batch) * edges[k][kk].to("cpu")
                            for kk in edges[k].keys()
                        }
                        for k in edges.keys()
                    }
            else:
                for k in nodes.keys():
                    if k != "y":
                        running_nodes[k] += len(batch) * nodes[k].to("cpu")
                if not args.nodes_only:
                    for k in edges.keys():
                        for v in edges[k].keys():
                            running_edges[k][v] += len(batch) * edges[k][v].to("cpu")

            # memory cleanup
            del nodes, edges
            gc.collect()

        nodes = {k: v.to(device) / num_examples for k, v in running_nodes.items()}
        if not args.nodes_only:
            edges = {
                k: {
                    kk: 1 / num_examples * v.to(device)
                    for kk, v in running_edges[k].items()
                }
                for k in running_edges.keys()
            }
        else:
            edges = None

        save_dict = {"examples": examples, "nodes": nodes, "edges": edges}
        # Create the full path for saving the circuit file
        save_path = os.path.join(args.circuit_dir, f"{save_base}_{node_suffix}.pt")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, "wb") as outfile:
            t.save(save_dict, outfile)

    # feature annotations
    if os.path.exists(
        annotations_path := f"annotations/{args.model.split('/')[-1]}.jsonl"
    ):
        print(f"Loading feature annotations from {annotations_path}")
        annotations = {}
        with open(annotations_path, "r") as f:
            for line in f:
                line = json.loads(line)
                if "Annotation" in line:
                    annotations[line["Name"]] = line["Annotation"]
    else:
        annotations = None

    if args.aggregation == "none":
        example = examples[0]["clean_prefix"]
        plot_save_path = f"{args.plot_dir}/{save_base}_node{args.node_threshold}_edge{args.edge_threshold}"
        os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
        plot_circuit_posaligned(
            nodes,
            edges,
            layers=n_layers,
            example_text=example,
            node_threshold=args.node_threshold,
            edge_threshold=args.edge_threshold,
            pen_thickness=args.pen_thickness,
            annotations=annotations,
            save_dir=plot_save_path,
            gemma_mode=(args.model == "google/gemma-2-2b"),
            parallel_attn=parallel_attn,
        )
    else:
        plot_save_path = f"{args.plot_dir}/{save_base}_node{args.node_threshold}_edge{args.edge_threshold}_n{num_examples}_agg{args.aggregation}"
        os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
        plot_circuit(
            nodes,
            edges,
            layers=n_layers,
            node_threshold=args.node_threshold,
            edge_threshold=args.edge_threshold,
            pen_thickness=args.pen_thickness,
            annotations=annotations,
            save_dir=plot_save_path,
            gemma_mode=(args.model == "google/gemma-2-2b"),
            parallel_attn=parallel_attn,
        )
