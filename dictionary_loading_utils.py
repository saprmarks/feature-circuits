from re import split
from collections import namedtuple
from dictionary_learning import AutoEncoder, JumpReluAutoEncoder, IdentityDict
from attribution import Submodule
from typing import Literal
import torch as t
from huggingface_hub import list_repo_files
from tqdm import tqdm
import os
DICT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/dictionaries"

DictionaryStash = namedtuple("DictionaryStash", ["embed", "attns", "mlps", "resids"])

def _load_pythia_saes_and_submodules(
    model,
    thru_layer: int | None = None,
    separate_by_type: bool = False,
    include_embed: bool = True,
    neurons: bool = False,
    dtype: t.dtype = t.float32,
    device: t.device = t.device("cpu"),
):
    assert len(model.gpt_neox.layers) == 6, "Not the expected number of layers for pythia-70m-deduped"
    if thru_layer is None:
        thru_layer = len(model.gpt_neox.layers)

    attns = []
    mlps = []
    resids = []
    dictionaries = {}
    if include_embed:
        embed = Submodule(
            name = "embed",
            submodule=model.gpt_neox.embed_in,
        )
        if not neurons:
            dictionaries[embed] = AutoEncoder.from_pretrained(
                f"{DICT_DIR}/pythia-70m-deduped/embed/10_32768/ae.pt",
                dtype=dtype,
                device=device,
            )
        else:
            dictionaries[embed] = IdentityDict(512)
    else:
        embed = None
    for i, layer in enumerate(model.gpt_neox.layers[:thru_layer+1]):
        attns.append(
            attn := Submodule(
                name = f"attn_{i}",
                submodule=layer.attention,
                is_tuple=True,
            )
        )
        mlps.append(
            mlp := Submodule(
                name = f"mlp_{i}",
                submodule=layer.mlp,
            )
        )
        resids.append(
            resid := Submodule(
                name = f"resid_{i}",
                submodule=layer,
                is_tuple=True,
            )
        )
        if not neurons:
            dictionaries[attn] = AutoEncoder.from_pretrained(
                f"{DICT_DIR}/pythia-70m-deduped/attn_out_layer{i}/10_32768/ae.pt",
                dtype=dtype,
                device=device,
            )
            dictionaries[mlp] = AutoEncoder.from_pretrained(
                f"{DICT_DIR}/pythia-70m-deduped/mlp_out_layer{i}/10_32768/ae.pt",
                dtype=dtype,
                device=device,
            )
            dictionaries[resid] = AutoEncoder.from_pretrained(
                f"{DICT_DIR}/pythia-70m-deduped/resid_out_layer{i}/10_32768/ae.pt",
                dtype=dtype,
                device=device,
            )
        else:
            dictionaries[attn] = IdentityDict(512)
            dictionaries[mlp] = IdentityDict(512)
            dictionaries[resid] = IdentityDict(512)

    if separate_by_type:
        return DictionaryStash(embed, attns, mlps, resids), dictionaries
    else:
        submodules = (
            [embed] if include_embed else []
         ) + [
            x for layer_dictionaries in zip(attns, mlps, resids) for x in layer_dictionaries
        ]
        return submodules, dictionaries

def load_gemma_sae(
    submod_type: Literal["embed", "attn", "mlp", "resid"],
    layer: int,
    width: Literal["16k", "65k"] = "16k",
    neurons: bool = False,
    dtype: t.dtype = t.float32,
    device: t.device = t.device("cpu"),
):
    if neurons:
        if submod_type != "attn":
            return IdentityDict(2304)
        else:
            return IdentityDict(2048)

    repo_id = "google/gemma-scope-2b-pt-" + (
        "res" if submod_type in ["embed", "resid"] else
        "att" if submod_type == "attn" else
        "mlp"
    )
    if submod_type != "embed":
        directory_path = f"layer_{layer}/width_{width}"
    else:
        directory_path = "embedding/width_4k"

    files_with_l0s = [
        (f, int(f.split("_")[-1].split("/")[0]))
        for f in list_repo_files(repo_id, repo_type="model", revision="main")
        if f.startswith(directory_path) and f.endswith("params.npz")
    ]
    optimal_file = min(files_with_l0s, key=lambda x: abs(x[1] - 100))[0]
    optimal_file = optimal_file.split("/params.npz")[0]
    return JumpReluAutoEncoder.from_pretrained(
        load_from_sae_lens=True,
        release=repo_id.split("google/")[-1],
        sae_id=optimal_file,
        dtype=dtype,
        device=device,
    )

def _load_gemma_saes_and_submodules(
    model,
    thru_layer: int | None = None,
    separate_by_type: bool = False,
    include_embed: bool = True,
    neurons: bool = False,
    dtype: t.dtype = t.float32,
    device: t.device = t.device("cpu"),
):
    assert len(model.model.layers) == 26, "Not the expected number of layers for Gemma-2-2B"
    if thru_layer is None:
        thru_layer = len(model.model.layers)
    
    attns = []
    mlps = []
    resids = []
    dictionaries = {}
    if include_embed:
        embed = Submodule(
            name = "embed",
            submodule=model.model.embed_tokens,
        )
        dictionaries[embed] = load_gemma_sae("embed", 0, neurons=neurons, dtype=dtype, device=device)
    else:
        embed = None
    for i, layer in tqdm(enumerate(model.model.layers[:thru_layer+1]), total=thru_layer+1, desc="Loading Gemma SAEs"):
        attns.append(
            attn := Submodule(
                name=f"attn_{i}",
                submodule=layer.self_attn.o_proj,
                use_input=True
            )
        )
        dictionaries[attn] = load_gemma_sae("attn", i, neurons=neurons, dtype=dtype, device=device)
        mlps.append(
            mlp := Submodule(
                name=f"mlp_{i}",
                submodule=layer.post_feedforward_layernorm,
            )
        )
        dictionaries[mlp] = load_gemma_sae("mlp", i, neurons=neurons, dtype=dtype, device=device)
        resids.append(
            resid := Submodule(
                name=f"resid_{i}",
                submodule=layer,
                is_tuple=True,
            )
        )
        dictionaries[resid] = load_gemma_sae("resid", i, neurons=neurons, dtype=dtype, device=device)

    if separate_by_type:
        return DictionaryStash(embed, attns, mlps, resids), dictionaries
    else:
        submodules = (
            [embed] if include_embed else []
        )+ [
            x for layer_dictionaries in zip(attns, mlps, resids) for x in layer_dictionaries
        ]
        return submodules, dictionaries


def load_saes_and_submodules(
    model,
    model_name: Literal["EleutherAI/pythia-70m-deduped", "google/gemma-2-2b"],
    thru_layer: int | None = None,
    separate_by_type: bool = False,
    include_embed: bool = True,
    neurons: bool = False,
    dtype: t.dtype = t.float32,
    device: t.device = t.device("cpu"),
):
    if model_name == "EleutherAI/pythia-70m-deduped":
        return _load_pythia_saes_and_submodules(model, thru_layer=thru_layer, separate_by_type=separate_by_type, include_embed=include_embed, neurons=neurons, dtype=dtype, device=device)
    elif model_name == "google/gemma-2-2b":
        return _load_gemma_saes_and_submodules(model, thru_layer=thru_layer, separate_by_type=separate_by_type, include_embed=include_embed, neurons=neurons, dtype=dtype, device=device)
    else:
        raise ValueError(f"Model {model_name} not supported")