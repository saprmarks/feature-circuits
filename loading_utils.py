from dataclasses import dataclass
import torch as t
from nnsight.envoy import Envoy
from collections import namedtuple
from dictionary_learning import AutoEncoder, JumpReluAutoEncoder
from dictionary_learning.dictionary import IdentityDict
from typing import Literal
from huggingface_hub import list_repo_files
from tqdm import tqdm
import os

DICT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/dictionaries"

@dataclass(frozen=True)
class Submodule:
    name: str
    submodule: Envoy
    use_input: bool = False
    is_tuple: bool = False

    def __hash__(self):
        return hash(self.name)

    def get_activation(self):
        if self.use_input:
            out = self.submodule.input # TODO make sure I didn't break for pythia
        else:
            out = self.submodule.output
        if self.is_tuple:
            return out[0]
        else:
            return out

    def set_activation(self, x):
        if self.use_input:
            if self.is_tuple:
                self.submodule.input[0][:] = x
            else:
                self.submodule.input[:] = x
        else:
            if self.is_tuple:
                self.submodule.output[0][:] = x
            else:
                self.submodule.output[:] = x

    def stop_grad(self):
        if self.use_input:
            if self.is_tuple:
                self.submodule.input[0].grad = t.zeros_like(self.submodule.input[0])
            else:
                self.submodule.input.grad = t.zeros_like(self.submodule.input)
        else:
            if self.is_tuple:
                self.submodule.output[0].grad = t.zeros_like(self.submodule.output[0])
            else:
                self.submodule.output.grad = t.zeros_like(self.submodule.output)


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
    thru_layer: int | None = None,
    separate_by_type: bool = False,
    include_embed: bool = True,
    neurons: bool = False,
    dtype: t.dtype = t.float32,
    device: t.device = t.device("cpu"),
):
    model_name = model.config._name_or_path

    if model_name == "EleutherAI/pythia-70m-deduped":
        return _load_pythia_saes_and_submodules(model, thru_layer=thru_layer, separate_by_type=separate_by_type, include_embed=include_embed, neurons=neurons, dtype=dtype, device=device)
    elif model_name == "google/gemma-2-2b":
        return _load_gemma_saes_and_submodules(model, thru_layer=thru_layer, separate_by_type=separate_by_type, include_embed=include_embed, neurons=neurons, dtype=dtype, device=device)
    else:
        raise ValueError(f"Model {model_name} not supported")
    
#### TESTING ####

def test_submodule_shapes(model, submodule):
    """Test that activations and gradients have consistent shapes [batch, seq_len, dim]"""
    # Get baseline output
    with model.trace("test input"):
        out1 = model.output.logits.sum().save()
    
    # Test gradient shape in separate trace
    with model.trace("test input"):
        acts = submodule.get_activation().save()
        grads = submodule.get_activation().grad.save()
        out2 = model.output.logits.sum().save()
        out2.backward()

    assert t.allclose(out1.value, out2.value), f"Submodule {submodule.name}: Output should be the same before and after grabbing acts/grads"
    assert len(acts.shape) == 3, f"Submodule {submodule.name}: Expected 3D tensor [batch, seq_len, dim], got shape {acts.shape}"
    assert grads.shape == acts.shape, f"Submodule {submodule.name}: Gradient shape {grads.shape} doesn't match activation shape {acts.shape}"

def test_activation_intervention(model, submodule):
    """Test that zeroing activations changes the output"""
    # Get baseline output
    with model.trace("test input"):
        acts = submodule.get_activation().save()
        out1 = model.output.logits.sum().save()
    
    # Test intervention in separate trace
    with model.trace("test input"):
        submodule.set_activation(t.randn_like(acts.value))
        out2 = model.output.logits.sum().save()
    
    assert not t.allclose(out1.value, out2.value), f"Submodule {submodule.name}: Output should change after modifying activation"

def test_gradient_intervention(model, upstream_submodule, downstream_submodule):
    """Test that stopping gradients affects upstream gradients"""
    # Get baseline gradients
    with model.trace("test input"):
        grads1 = upstream_submodule.get_activation().grad.save()
        out1 = model.output.logits.sum().save()
        out1.backward()
    
    # Test intervention
    with model.trace("test input"):
        grads2 = upstream_submodule.get_activation().grad.save()
        downstream_submodule.stop_grad()
        out2 = model.output.logits.sum().save()
        out2.backward()
    
    assert t.allclose(out1.value, out2.value), f"Submodule {upstream_submodule.name}: Output should be same before and after stopping gradients"
    assert not t.allclose(grads1.value, grads2.value), f"Submodule {upstream_submodule.name}: Gradients should change after stopping grad flow in {downstream_submodule.name}"

def run_tests(model):
    """Run all tests for the Submodule class"""
    submodules, _ = load_saes_and_submodules(model, neurons=True, include_embed=False)  # Use neurons=True to avoid loading dictionaries
    
    # Test shapes on different types of submodules
    print("Testing submodule shapes...")
    test_cases = [
        submodules[0],  # embed
        submodules[1],  # first attn
        submodules[2],  # first mlp
        submodules[3],  # first resid
        submodules[-3], # last mlp
        submodules[-1], # last resid
    ]
    for submodule in test_cases:
        print(f"Testing shapes for {submodule.name}...")
        test_submodule_shapes(model, submodule)
    print("Shape tests passed!")
    
    # Test activation interventions
    print("\nTesting activation interventions...")
    for submodule in test_cases:
        print(f"Testing activation intervention for {submodule.name}...")
        test_activation_intervention(model, submodule)
    print("Activation intervention tests passed!")
    
    # Test gradient interventions
    print("\nTesting gradient interventions...")
    gradient_test_pairs = [
        (submodules[0], submodules[2]),   # attn_0 -> resid_0
        (submodules[1], submodules[2]),   # mlp_0 -> resid_0
        (submodules[-2], submodules[-1]), # last mlp -> last resid
    ]
    for upstream, downstream in gradient_test_pairs:
        print(f"Testing gradient intervention from {upstream.name} to {downstream.name}...")
        test_gradient_intervention(model, upstream, downstream)
    print("Gradient intervention tests passed!")
    
    print("\nAll tests passed!")

if __name__ == "__main__":
    from nnsight import LanguageModel
    
    print("Loading pythia-70m-deduped...")
    model = LanguageModel("EleutherAI/pythia-70m-deduped", dispatch=True, device_map="auto")
    
    print("Running tests for pythia-70m-deduped")
    run_tests(model)

    print("Loading model gemma-2-2b...")
    model = LanguageModel("google/gemma-2-2b", dispatch=True, device_map="auto")

    print("Running tests for gemma-2-2b")
    run_tests(model)



