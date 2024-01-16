from tqdm import tqdm

import torch as t
from loading_utils import load_submodule
from dictionary_learning.dictionary import AutoEncoder

def run_with_ablated_features(model, example, dictionary_dir, dictionary_size, features,
                              return_submodules=None):
    """
    TODO: This method currently uses zero ablations. Should update this to also support
    naturalistic ablations.
    """
    # usage: features_per_layer[layer][submodule_type]
    features_per_layer = defaultdict(lambda: defaultdict(list))
    for feature in features:
        submodule_type, layer_and_feat_idx = feature.split("_")
        layer, feat_idx = layer_and_feat_idx.split("/")
        features_per_layer[int(layer)][submodule_type].append(int(feat_idx))

    saved_submodules = {}
    def _ablate_features(submodule_type, layer, feature_list):
        submodule_name = submodule_type_to_name(submodule_type).format(layer)
        submodule = load_submodule(model, submodule_name)
        # if there are no features to ablate, just return the submodule output
        if len(feature_list) == 0 and return_submodules and submodule_name in return_submodules:
            saved_submodules[submodule_name] = submodule.output.save()
            return
        
        # load autoencoder
        is_resid = len(submodule.output[0].shape) > 2
        if is_resid:
            submodule_width = submodule.output[0].shape[2]
        else:
            submodule_width = submodule.out_features
        autoencoder = AutoEncoder(submodule_width, dictionary_size).cuda()
        try:
            autoencoder.load_state_dict(
                t.load(os.path.join(dictionary_dir, f"{submodule_type}_out_layer{layer}/1_32768/ae.pt"))
            )
        except FileNotFoundError:
            autoencoder.load_state_dict(
                t.load(os.path.join(dictionary_dir, f"{submodule_type}_out_layer{layer}/0_32768/ae.pt"))
            )

        # encode activations into features
        if is_resid:
            x = submodule.output[0]
        else:
            x = submodule.output
        f = autoencoder.encode(x)
        for feature_idx in feature_list:
            f[:, :, feature_idx] = 0.0      # ablate features
        # replace submodule w/ autoencoder out
        if is_resid:
            submodule.output[0] = autoencoder.decode(f)
        else:
            submodule.output = autoencoder.decode(f)

        # replace activations of submodule
        if return_submodules and submodule_name in return_submodules:
            saved_submodules[submodule_name] = submodule.output.save()

    with model.invoke(example) as invoker:
        # from lowest layer to highest (does this matter in nnsight?)
        for layer in sorted(features_per_layer):
            for submodule_type in features_per_layer[layer]:
                _ablate_features(submodule_type, layer, features_per_layer[layer][submodule_type])
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