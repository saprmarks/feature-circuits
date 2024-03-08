import os
from tqdm import tqdm

import torch as t
from loading_utils import load_submodule, submodule_type_to_name
from collections import defaultdict
from dictionary_learning.dictionary import AutoEncoder

def run_with_ablated_features(model, example, dictionary_size, features, dictionaries,
                              return_submodules=None, inverse=False, patch_vector=None):
    """
    TODO: This method currently uses zero ablations. Should update this to also support
    naturalistic ablations.
    Setting `inverse = True` ablates everything EXCEPT the provided features.
    """
    # usage: features_per_layer[layer][submodule_type]
    features_per_layer = defaultdict(lambda: defaultdict(list))
    saved_submodules = {}

    # add all submodules to features_per_layer (necessary for inverse)
    for layer in range(model.config.num_hidden_layers):
        features_per_layer[layer]["mlp"] = []
        features_per_layer[layer]["attn"] = []
        features_per_layer[layer]["resid"] = []
    # add feature indices to features_per_layer
    for feature in features:
        submodule_type, layer_and_feat_idx = feature.split("_")
        layer, feat_idx = layer_and_feat_idx.split("/")
        features_per_layer[int(layer)][submodule_type].append(int(feat_idx))

    def _ablate_features(submodule_type, layer, feature_list, patch_vector):
        submodule_name = submodule_type_to_name(submodule_type).format(layer)
        is_resid = "mlp" not in submodule_name and "attention" not in submodule_name
        submodule = load_submodule(model, submodule_name)

        # if there are no features to ablate, just return the submodule output
        if len(feature_list) == 0 and return_submodules and submodule_name in return_submodules:
            saved_submodules[submodule_name] = submodule.output.save()
            return

        # load autoencoder
        autoencoder = dictionaries[submodule]

        if patch_vector is None:
            patch_vector = t.zeros(dictionary_size)     # do zero ablation

        # encode activations into features
        if is_resid:
            x = submodule.output[0]
        else:
            x = submodule.output
        f = autoencoder.encode(x)

        if inverse:
            patch_copy = patch_vector.clone().repeat(f.shape[1], 1)
            patch_copy[:, feature_list] = f[:, :, feature_list]
            f[:, :, feature_list] = patch_copy[:, feature_list]                  # ablate f everywhere except feature indices
        else:
            f[:, :, feature_list] = patch_vector[feature_list]   # ablate f at feature indices
        # replace submodule w/ autoencoder out
        if is_resid:
            submodule.output = (autoencoder.decode(f), *submodule.output[1:])
        else:
            submodule.output = autoencoder.decode(f)

        # replace activations of submodule
        if return_submodules and submodule_name in return_submodules:
            saved_submodules[submodule_name] = submodule.output.save()

    with model.invoke(example) as invoker:
        # from lowest layer to highest (does this matter in nnsight?)
        for layer in sorted(features_per_layer):
            for submodule_type in features_per_layer[layer]:
                _ablate_features(submodule_type, layer, features_per_layer[layer][submodule_type], patch_vector)
        if return_submodules:
            for submodule_name in return_submodules:
                if submodule_name not in saved_submodules.keys():
                    print(submodule_name)
                    submodule_type = "mlp" if "mlp" in submodule_name else "attn" if "attention" in submodule_name else "resid"
                    submodule_parts = submodule_name.split(".")
                    for idx, part in enumerate(submodule_parts):
                        if part.startswith("layer"):
                            layer = int(submodule_parts[idx+1])
                            break
                    _ablate_features(submodule_type, layer, [])
    saved_submodules["model"] = invoker.output

    return saved_submodules


def get_mean_activations(model, buffer, submodule, autoencoder, steps=5000, device="cuda"):
    """
    For use in mean ablation.
    For random ablation, just set `steps` = 1.
    """
    f_mean = t.zeros(autoencoder.encoder.out_features).to(device)
    x_hat_mean = t.zeros(autoencoder.encoder.in_features).to(device)

    for step, submodule_acts in enumerate(tqdm(buffer, total=steps)):
        if step >= steps:
            break

        if isinstance(submodule_acts, t.Tensor):
            submodule_acts = submodule_acts.to(device)
            in_acts = out_acts = submodule_acts
        else:
            submodule_acts = tuple(a.to(device) for a in submodule_acts)
            in_acts, out_acts = submodule_acts
        

        f = autoencoder.encode(in_acts)
        f_mean += t.sum(f, dim=0)
        x_hat = autoencoder.decode(f)
        x_hat_mean += t.sum(x_hat, dim=0)
    
    f_mean /= steps
    x_hat_mean /= steps

    return {"mean_encoded": f_mean,
            "mean_decoded": x_hat_mean}


if __name__ == "__main__":
    import zstandard
    import json
    import io
    import sys
    import pickle
    import os
    from nnsight import LanguageModel
    from dictionary_learning.buffer import ActivationBuffer

    submodule_name = sys.argv[1]
    ae_path = sys.argv[2]

    model = LanguageModel("EleutherAI/pythia-70m-deduped", device_map="cuda:0")
    submodule = load_submodule(model, submodule_name)
    autoencoder = AutoEncoder(512, 32768).to("cuda")
    # autoencoder.load_state_dict(t.load("/share/projects/dictionary_circuits/autoencoders/pythia-70m-deduped/mlp_out_layer4/1_32768/ae.pt"))
    autoencoder.load_state_dict(t.load(ae_path))

    data_path = '/share/data/datasets/pile/the-eye.eu/public/AI/pile/train/00.jsonl.zst'
    compressed_file = open(data_path, 'rb')
    dctx = zstandard.ZstdDecompressor()
    reader = dctx.stream_reader(compressed_file)
    text_stream = io.TextIOWrapper(reader, encoding='utf-8')
    def generator():
        for line in text_stream:
            yield json.loads(line)['text']
    data = generator()
    buffer = ActivationBuffer(
        data,
        model,
        submodule,
        io='out',
        in_feats=512,
        out_feats=512,
        in_batch_size=40,
        out_batch_size=512 * 40,
        n_ctxs=40 * 100,
    )

    with t.no_grad():
        mean_acts = get_mean_activations(model, buffer, submodule, autoencoder, steps=10000)

    print("Non-zero features:", t.count_nonzero(mean_acts["mean_encoded"]))
    print("Non-zero autoencoded activations:", t.count_nonzero(mean_acts["mean_decoded"]))
    
    # write to file
    outpath = os.path.dirname(ae_path)
    outpath = os.path.join(outpath, "mean_acts.pkl")
    outdir = os.path.dirname(outpath)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    with open(outpath, "wb") as handle:
        pickle.dump(mean_acts, handle)