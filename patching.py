# %%
from subnetwork import Subnetwork, Node

def subnetwork_patch(
        clean,
        patch,
        model,
        submodules,
        autoencoders,
        subnetwork,
        node,
):
    if type(clean) == tuple:
        clean_input, clean_answer = clean
    else:
        clean_input = clean
    if type(patch) == tuple:
        patch_input, patch_answer = patch
    else:
        patch_input = patch

    # patch run
    with model.invoke(patch_input):
        for submodule, ae in zip(submodules, autoencoders):
            f = ae.encode(submodule.output)
            submodule.output = ae.decode(f)
            if node.submodule == submodule:
                f_patch = f.save()

    # clean run
    clean_features = []
    with model.invoke(clean_input) as invoker:
        for submodule, ae in zip(submodules, autoencoders):
            f = ae.encode(submodule.output)
            clean_features.append(f.save())
            submodule.output = ae.decode(f)

    if type(clean) == tuple:
        clean_idx = model.tokenizer(clean_answer)['input_ids'][-1]
        patch_idx = model.tokenizer(patch_answer)['input_ids'][-1]
        clean_diff = invoker.output.logits[0, -1, patch_idx] - invoker.output.logits[0, -1, clean_idx]
    # else:
    #     clean_out = invoker.output.logits[0, -1]

    clean_features = [f for f in clean_features]
    
    # subnetwork patched run
    with model.invoke(clean_input) as invoker:
        for submodule, ae, f_clean in zip(submodules, autoencoders, clean_features):
            f = ae.encode(submodule.output)

            # if this is the node to patch
            if node.submodule == submodule:
                if node.feature is None:
                    f = f_patch.value
                else:
                    f[:, :, node.feature] = f_patch.value[:, :, node.feature]
            
            # if node is not in subnetwork, patch back to clean
            if submodule not in subnetwork.whitelist:
                retain_idxs = t.ones(f.shape[-1], dtype=t.bool)
            elif subnetwork.whitelist[submodule] is None:
                retain_idxs = t.zeros(f.shape[-1], dtype=t.bool)
            else:
                retain_idxs = t.ones(f.shape[-1], dtype=t.bool)
                for feat_idx in subnetwork.whitelist[submodule]:
                    retain_idxs[feat_idx] = False
            f[:, :, :] = t.where(
                retain_idxs,
                f_clean.value,
                f
            )
                
            submodule.output = ae.decode(f)
    if type(patch) == tuple:
        clean_idx = model.tokenizer(clean_answer)['input_ids'][-1]
        patch_idx = model.tokenizer(patch_answer)['input_ids'][-1]
        patch_diff = invoker.output.logits[0, -1, patch_idx] - invoker.output.logits[0, -1, clean_idx]
    # else:
    #     patch_out = invoker.output.logits[0, -1]
    
    return patch_diff - clean_diff
# %%
from nnsight import LanguageModel
from dictionary_learning.dictionary import AutoEncoder
import torch as t

model = LanguageModel('EleutherAI/pythia-70m-deduped', device_map='cuda:0')
layers = len(model.gpt_neox.layers)

submodules = [
    model.gpt_neox.layers[i].mlp for i in range(layers)
]

autoencoders = []
for i in range(layers):
    ae = AutoEncoder(512, 64 * 512)
    ae.load_state_dict(t.load(f'/share/projects/dictionary_circuits/autoencoders/pythia-70m-deduped/mlp_out_layer{i}/1_32768/ae.pt'))
    autoencoders.append(ae)

clean = (
    'The men', 'are'
)
patch = (
    'The man', 'is'
)

node = Node(model, 'gpt_neox.layers.1.mlp/16149')

subnetwork = Subnetwork([
    Node(model, 'gpt_neox.layers.0.mlp'),
    Node(model, 'gpt_neox.layers.1.mlp'),
    Node(model, 'gpt_neox.layers.2.mlp'),
    Node(model, 'gpt_neox.layers.3.mlp'),
    Node(model, 'gpt_neox.layers.4.mlp'),
    Node(model, 'gpt_neox.layers.5.mlp'),
])


subnetwork_patch(
    clean,
    patch,
    model,
    submodules,
    autoencoders,
    subnetwork,
    node,
)
# %%
