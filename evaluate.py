import random
from dictionary_learning.dictionary import AutoEncoder, IdentityDict
from acdc import patching_on_y, patching_on_downstream_feature
from ablation_utils import run_with_ablated_features
from loading_utils import load_submodule_and_dictionary, DictionaryCfg
from tqdm import tqdm
from collections import defaultdict
from copy import deepcopy
import torch as t

def _normalize_name(name):
    layer, feat_idx, submodule_type = name.split("_")
    return f"{submodule_type}_{layer}/{feat_idx}"


def evaluate_faithfulness(circuit_features, model, dict_cfg, eval_dataset, dictionaries,
                          patch_type='zero', mean_vectors=None, include_errors=None):
    """
    Evaluate performance of circuit compared to full model.
    `patch_type` can be one of the following:
    - "zero": sets activation to zero
    - "mean": sets activation to its mean over many Pile contexts (loads from .pkl)
    - "random": sets activation to what it would've been given a single Pile
                context (computed in-function)
    """
    if patch_type == "random":
        raise NotImplementedError()
    
    mean_faithfulness = 0.
    num_examples = len(eval_dataset)
    faithfulness_by_example = {}
    for example in tqdm(eval_dataset, desc="Evaluating faithfulness", total=len(eval_dataset)):
        with model.trace(example["clean_prefix"]):
            model_out = model.embed_out.output.save()
        model_logit_diff = model_out.value[:,-1,example["clean_answer"]] - \
                           model_out.value[:,-1,example["patch_answer"]]
        # model_out = run_with_ablated_features(model, example["clean_prefix"], dict_cfg.size, 
        #                                       [], dictionaries,
        #                                       patch_type=patch_type, mean_vectors=mean_vectors,
        #                                       inverse=False)["model"]
        # model_logit_diff = model_out[:, -1, example["clean_answer"]] - \
        #                     model_out[:, -1, example["patch_answer"]]
        circuit_out = run_with_ablated_features(model, example["clean_prefix"], dict_cfg.size,
                                                circuit_features, dictionaries,
                                                patch_type=patch_type, mean_vectors=mean_vectors,
                                                include_errors=include_errors,
                                                inverse=True)["model"]
        circuit_logit_diff = circuit_out[:, -1, example["clean_answer"]] - \
                                circuit_out[:, -1, example["patch_answer"]]

        faithfulness = circuit_logit_diff / model_logit_diff
        mean_faithfulness += faithfulness
    
    # sorted_faithfulness = {k: v for k, v in sorted(faithfulness_by_example.items(), key=lambda x: x[1])}
    # for example in sorted_faithfulness:
    #     print(f"{example}: {sorted_faithfulness[example]}")

    mean_faithfulness /= num_examples
    return mean_faithfulness.item()


def evaluate_completeness(circuit_features, model, dict_cfg, eval_dataset, dictionaries,
                          patch_type='zero', mean_vectors=None, include_errors=None):
    """
    Evaluate whether we've found everything contributing to the logit diff.
    `patch_type` can be one of the following:
    - "zero": sets activation to zero
    - "mean": sets activation to its mean over many Pile contexts (loads from .pkl)
    - "random": sets activation to what it would've been given a single Pile
                context (computed in-function)
    """
    circuit_feature_set = set(circuit_features)

    if patch_type == "random":
        raise NotImplementedError()
    
    mean_percent_recovered = 0
    completeness_points = []
    mean_incompleteness = 0.
    total = 0
    K = circuit_feature_set
    num_examples = len(eval_dataset)

    # compute incompleteness
    model_no_K_diff = 0.
    circuit_features_no_K = circuit_feature_set.difference(K)
    completeness = 0.
    baseline = 0.
    for example in tqdm(eval_dataset, desc="Evaluating completeness", total=len(eval_dataset)):
        model_no_K_out = run_with_ablated_features(model, example["clean_prefix"],
                                    dict_cfg.size,
                                    list(K), dictionaries,
                                    patch_type=patch_type, mean_vectors=mean_vectors,
                                    include_errors=include_errors,
                                    inverse=False)["model"]
        model_no_K_diff = model_no_K_out[:, -1, example["clean_answer"]] - \
                            model_no_K_out[:, -1, example["patch_answer"]]
        circuit_no_K_out = run_with_ablated_features(model, example["clean_prefix"],
                                                        dict_cfg.size,
                                                        list(circuit_features_no_K), dictionaries,
                                                        patch_type=patch_type, mean_vectors=mean_vectors,
                                                        include_errors=include_errors,
                                                        inverse=True)["model"]
        circuit_no_K_diff = circuit_no_K_out[:, -1, example["clean_answer"]] - \
                            circuit_no_K_out[:, -1, example["patch_answer"]]
        # baseline: how well does the full model do compared to the empty set?
        with model.trace(example["clean_prefix"]), t.inference_mode():
            model_diff = model.embed_out.output[:,-1,example["clean_answer"]] - \
                         model.embed_out.output[:,-1,example["patch_answer"]]
            model_diff = model_diff.save()
        model_diff = model_diff.value

        completeness += circuit_no_K_diff / model_no_K_diff
        baseline += circuit_no_K_diff / model_diff
    
    completeness /= num_examples
    baseline /= num_examples
    return {"mean_completeness": completeness.item(),
            "baseline": baseline.item(),
            "K": K}


def evaluate_minimality(circuit_features):
    return -len(circuit_features)


def load_triangles_circuit(circuit_path, n_layers):
    with open(circuit_path, "rb") as handler:
        circuit = t.load(handler)
    
    node_threshold = float(circuit_path.split("node")[1].split("_")[0])
    include_errors = {}

    circuit_nodes = circuit["nodes"]
    feature_list = []
    errors = {}
    
    if "embed" in circuit_nodes:
        idxs = (circuit_nodes["embed"].act.abs() > node_threshold).nonzero()
        for idx in idxs:
            if len(idx) == 1:
                feature_list.append(f"resid_-1/{idx[0]}")
            else:   # we have sequence positions
                feature_list.append(f"{idx[0]}, resid_-1/{idx[1]}")
        
        # scalar if no sequence positions. list if sequence positions
        include_errors[model.gpt_neox.embed_in] = circuit_nodes["embed"].resc.abs() > node_threshold
    else:
        circuit_nodes["embed"] = []
        include_errors[model.gpt_neox.embed_in] = t.Tensor([False])

    for submodtype in ["mlp", "attn", "resid"]:
        for layer in range(n_layers):
            submodname = f"{submodtype}_{layer}"
            submodule = f"model.gpt_neox.layers[{layer}]"
            if submodtype == "mlp":
                submodule += ".mlp"
            elif submodtype == "attn":
                submodule += ".attention"
            submodule = eval(submodule)

            include_errors[submodule] = circuit_nodes[submodname].resc.abs() > node_threshold
            # not clear whether we should include counterproductive features in these measures
            idxs = (circuit_nodes[submodname].act.abs() > node_threshold).nonzero()
            for idx in idxs:
                if len(idx) == 1:
                    feature_list.append(f"{submodtype}_{layer}/{idx[0]}")
                else:
                    feature_list.append(f"{idx[0]}, {submodtype}_{layer}/{idx[1]}")

    return feature_list, include_errors


if __name__ == "__main__":
    import argparse
    from nnsight import LanguageModel
    from loading_utils import load_examples
    from dictionary_learning.utils import hf_dataset_to_generator

    parser = argparse.ArgumentParser()
    parser.add_argument("circuit_path", type=str)
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-70m-deduped")
    parser.add_argument("--dataset", type=str, default="/share/projects/dictionary_circuits/data/phenomena/simple_test.json")
    parser.add_argument("--dict_id", type=str, default="10")
    parser.add_argument("--num_examples", type=int, default=100)

    args = parser.parse_args()

    model = LanguageModel(args.model, device_map="cuda:0", dispatch=True)
    d_model = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    dataset = load_examples(args.dataset, args.num_examples, model)

    embed = model.gpt_neox.embed_in
    attns = [layer.attention for layer in model.gpt_neox.layers]
    mlps = [layer.mlp for layer in model.gpt_neox.layers]
    resids = [layer for layer in model.gpt_neox.layers]
    dictionaries = {}

    # compute mean vectors on an unrelated dataset
    mean_vectors = defaultdict(lambda: t.zeros(model.config.hidden_size).to("cuda:0"))
    corpus = hf_dataset_to_generator("monology/pile-uncopyrighted")
    for _ in range(2):
        text = model.tokenizer(next(corpus), return_tensors="pt",
                                max_length=128, padding=True, truncation=True)
        seq_len = text["input_ids"].shape[1]
        attn_acts, mlp_acts, resid_acts, embed_acts = {}, {}, {}, 0.
        with model.trace(text):
            token_pos = random.randint(0, seq_len-1)
            embed_acts = embed.output[0, token_pos, :].save()
            for i in range(len(model.gpt_neox.layers)):
                attn_acts[i] = attns[i].output[0][0, token_pos, :].save()
                mlp_acts[i] = mlps[i].output[0, token_pos, :].save()
                resid_acts[i] = resids[i].output[0][0, token_pos, :].save()
        
        mean_vectors[embed] += embed_acts.value
        for i in range(len(model.gpt_neox.layers)):
            mean_vectors[attns[i]] += attn_acts[i].value
            mean_vectors[mlps[i]] += mlp_acts[i].value
            mean_vectors[resids[i]] += resid_acts[i].value

    mean_vectors[embed] = mean_vectors[embed].div(100.)
    for i in range(len(model.gpt_neox.layers)):
        mean_vectors[attns[i]] = mean_vectors[attns[i]].div(100.)
        mean_vectors[mlps[i]] = mean_vectors[mlps[i]].div(100.)
        mean_vectors[resids[i]] = mean_vectors[resids[i]].div(100.)

    if args.dict_id == 'id':
        from dictionary_learning.dictionary import IdentityDict
        patch_type = "mean"
        dict_cfg = DictionaryCfg("/share/projects/dictionary_circuits/autoencoders/pythia-70m-deduped/",
                            512)
        dict_size = 512

        dictionaries[embed] = IdentityDict(model.config.hidden_size)
        for i in range(len(model.gpt_neox.layers)):
            dictionaries[attns[i]] = IdentityDict(model.config.hidden_size)
            dictionaries[mlps[i]] = IdentityDict(model.config.hidden_size)
            dictionaries[resids[i]] = IdentityDict(model.config.hidden_size)

    else:
        patch_type = "zero"
        dict_cfg = DictionaryCfg("/share/projects/dictionary_circuits/autoencoders/pythia-70m-deduped/",
                            32768)
        dict_size = 32768

        ae = AutoEncoder(d_model, dict_size).to("cuda:0")
        ae.load_state_dict(t.load(f'/share/projects/dictionary_circuits/autoencoders/pythia-70m-deduped/embed/{args.dict_id}_{dict_size}/ae.pt'))
        dictionaries[embed] = ae
        for i in range(len(model.gpt_neox.layers)):
            ae = AutoEncoder(d_model, dict_size).to("cuda:0")
            ae.load_state_dict(t.load(f'/share/projects/dictionary_circuits/autoencoders/pythia-70m-deduped/attn_out_layer{i}/{args.dict_id}_{dict_size}/ae.pt'))
            dictionaries[attns[i]] = ae

            ae = AutoEncoder(d_model, dict_size).to("cuda:0")
            ae.load_state_dict(t.load(f'/share/projects/dictionary_circuits/autoencoders/pythia-70m-deduped/mlp_out_layer{i}/{args.dict_id}_{dict_size}/ae.pt'))
            dictionaries[mlps[i]] = ae

            ae = AutoEncoder(d_model, dict_size).to("cuda:0")
            ae.load_state_dict(t.load(f'/share/projects/dictionary_circuits/autoencoders/pythia-70m-deduped/resid_out_layer{i}/{args.dict_id}_{dict_size}/ae.pt'))
            dictionaries[resids[i]] = ae

    circuit_features, include_errors = load_triangles_circuit(args.circuit_path, n_layers)

    faithfulness = evaluate_faithfulness(circuit_features, model, dict_cfg, dataset, dictionaries,
                                         patch_type=patch_type, mean_vectors=mean_vectors,
                                         include_errors=include_errors)
    completeness = evaluate_completeness(circuit_features, model, dict_cfg, dataset, dictionaries,
                                         patch_type=patch_type, mean_vectors=mean_vectors,
                                         include_errors=include_errors)
    minimality = evaluate_minimality(circuit_features)

    print("faithfulness (F(C) / F(M)):", faithfulness)
    print("completeness (F(empty) / F(M \ C)):", completeness["mean_completeness"])
    print("\tbaseline (F(empty) / F(M)):", completeness["baseline"])
    print("minimality (negative num nodes):", minimality)
