from nnsight import LanguageModel
import torch as t
from argparse import ArgumentParser
from activation_utils import SparseAct
from loading_utils import load_examples
from dictionary_loading_utils import load_saes_and_submodules


def run_with_ablations(
    clean,  # clean inputs
    patch,  # patch inputs for use in computing ablation values
    model,  # a nnsight LanguageModel
    submodules,  # list of submodules
    dictionaries,  # dictionaries[submodule] is an autoencoder for submodule's output
    nodes,  # nodes[submodule] is a boolean SparseAct with True for the nodes to keep (or ablate if complement is True)
    metric_fn,  # metric_fn(model, **metric_kwargs) -> t.Tensor
    metric_kwargs=dict(),
    complement=False,  # if True, then use the complement of nodes
    ablation_fn=lambda x: x.mean(dim=0).expand_as(
        x
    ),  # what to do to the patch hidden states to produce values for ablation, default mean ablation
    handle_errors="default",  # or 'remove' to zero ablate all; 'keep' to keep all
):
    if patch is None:
        patch = clean
    patch_states = {}
    with model.trace(patch), t.no_grad():
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            x = submodule.get_activation()
            f = dictionary.encode(x)
            x_hat = dictionary.decode(f)
            patch_states[submodule] = SparseAct(act=f, res=x - x_hat).save()
    patch_states = {k: ablation_fn(v.value) for k, v in patch_states.items()}

    with model.trace(clean), t.no_grad():
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            submod_nodes = nodes[submodule]
            x = submodule.get_activation()
            f = dictionary.encode(x)
            res = x - dictionary(x)

            # ablate features
            if complement:
                submod_nodes = ~submod_nodes
            submod_nodes.resc = submod_nodes.resc.expand(
                *submod_nodes.resc.shape[:-1], res.shape[-1]
            )
            if handle_errors == "remove":
                submod_nodes.resc = t.zeros_like(submod_nodes.resc).to(t.bool)
            if handle_errors == "keep":
                submod_nodes.resc = t.ones_like(submod_nodes.resc).to(t.bool)

            f[..., ~submod_nodes.act] = patch_states[submodule].act[
                ..., ~submod_nodes.act
            ]
            res[..., ~submod_nodes.resc] = patch_states[submodule].res[
                ..., ~submod_nodes.resc
            ]

            submodule.set_activation(dictionary.decode(f) + res)

        metric = metric_fn(model, **metric_kwargs).save()
    return metric.value


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="EleutherAI/pythia-70m-deduped",
        help="Name of model on which we evaluate faithfulness.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.1, help="Node threshold for the circuit."
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default="mean",
        help="Ablation style. Can be one of `mean`, `resample`, `zero`.",
    )
    parser.add_argument(
        "--circuit", type=str, required=True, help="Path to a circuit .pt file."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="rc_test.json",
        help="Data on which to evaluate the circuit.",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=100,
        help="Number of examples over which to evaluate the circuit.",
    )
    parser.add_argument(
        "--handle_errors",
        type=str,
        default="default",
        help="How to treat SAE error terms. Can be `default`, `keep`, or `remove`.",
    )
    parser.add_argument(
        "--start_layer",
        type=int,
        default=-1,
        help="Layer to evaluate the circuit from. Layers below --start_layer are given to the model for free.",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    dtype = {
        "EleutherAI/pythia-70m-deduped": t.float32,
        "google/gemma-2-2b": t.bfloat16,
    }[args.model]

    model = LanguageModel(
        args.model,
        attn_implementation="eager",
        torch_dtype=dtype,
        device_map=args.device,
        dispatch=True,
    )

    submodules, dictionaries = load_saes_and_submodules(
        model, include_embed=False, dtype=dtype, device=args.device
    )

    submodules = [
        s for s in submodules if int(s.name.split("_")[-1]) >= args.start_layer
    ]

    # Load circuit
    circuit = t.load(args.circuit)["nodes"]
    nodes = {
        submod: circuit[submod.name].abs() > args.threshold for submod in submodules
    }

    # Load examples
    examples = load_examples(
        f"data/{args.data}.json", args.examples, model, use_min_length_only=True
    )

    # Define ablation function
    if args.ablation == "resample":

        def ablation_fn(x):
            idxs = t.multinomial(
                t.ones(x.act.shape[0]), x.act.shape[0], replacement=True
            ).to(x.act.device)
            return SparseAct(act=x.act[idxs], res=x.res[idxs])
    elif args.ablation == "zero":
        def ablation_fn(x):
            return x.zeros_like()
    else:  # mean ablation
        def ablation_fn(x):
            return x.mean(dim=0).expand_as(x)

    # Prepare inputs
    clean_inputs = [e["clean_prefix"] for e in examples]
    clean_answer_idxs = t.tensor(
        [model.tokenizer(e["clean_answer"]).input_ids[-1] for e in examples],
        dtype=t.long,
        device=args.device,
    )
    patch_inputs = [e["patch_prefix"] for e in examples]
    patch_answer_idxs = t.tensor(
        [model.tokenizer(e["patch_answer"]).input_ids[-1] for e in examples],
        dtype=t.long,
        device=args.device,
    )

    def metric_fn(model):
        logits = model.output.logits[:, -1, :]
        return -t.gather(logits, dim=-1, index=patch_answer_idxs.view(-1, 1)).squeeze(
            -1
        ) + t.gather(logits, dim=-1, index=clean_answer_idxs.view(-1, 1)).squeeze(-1)

    # Compute faithfulness
    with t.no_grad():
        # Compute F(M)
        with model.trace(clean_inputs):
            metric = metric_fn(model).save()
        fm = metric.value.mean().item()

        # Compute F(C)
        fc = (
            run_with_ablations(
                clean_inputs,
                patch_inputs,
                model,
                submodules,
                dictionaries,
                nodes,
                metric_fn,
                ablation_fn=ablation_fn,
                handle_errors=args.handle_errors,
            )
            .mean()
            .item()
        )

        # Compute F(∅)
        fempty = (
            run_with_ablations(
                clean_inputs,
                patch_inputs,
                model,
                submodules,
                dictionaries,
                nodes={
                    submod: SparseAct(
                        act=t.zeros(dictionaries[submod].dict_size, dtype=t.bool),
                        resc=t.zeros(1, dtype=t.bool),
                    ).to(args.device)
                    for submod in submodules
                },
                metric_fn=metric_fn,
                ablation_fn=ablation_fn,
                handle_errors=args.handle_errors,
            )
            .mean()
            .item()
        )

    # Calculate faithfulness
    faithfulness = (fc - fempty) / (fm - fempty)

    print(f"Faithfulness: {faithfulness:.4f}")
    print(f"F(M): {fm:.4f}")
    print(f"F(C): {fc:.4f}")
    print(f"F(∅): {fempty:.4f}")
