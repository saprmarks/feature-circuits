import os
from tqdm import tqdm
from copy import copy

import torch as t
from loading_utils import load_submodule, submodule_type_to_name
from collections import defaultdict
from dictionary_learning.dictionary import AutoEncoder

def run_with_ablated_features(model, example, dictionary_size, features, dictionaries,
                              return_submodules=None, inverse=False, patch_type='zero',
                              mean_vectors=None, include_errors=None):
    """
    Setting `inverse = True` ablates everything EXCEPT the provided features.
    """
    # usage: features_per_layer[layer][submodule_type]
    features_per_layer = defaultdict(lambda: defaultdict(list))
    saved_submodules = defaultdict(list)

    # add all submodules to features_per_layer (necessary for inverse)
    features_per_layer[-1]["resid"] = []
    for layer in range(model.config.num_hidden_layers):
        features_per_layer[layer]["mlp"] = []
        features_per_layer[layer]["attn"] = []
        features_per_layer[layer]["resid"] = []

    # add feature indices to features_per_layer
    for feature in features:
        if "," in feature:
            seqpos, feature = feature.split(", ")
        else:
            seqpos = None
        submodule_type, layer_and_feat_idx = feature.split("_")
        layer, feat_idx = layer_and_feat_idx.split("/")

        if seqpos is not None:
            features_per_layer[int(layer)][submodule_type].append((int(seqpos), int(feat_idx)))
        else:
            features_per_layer[int(layer)][submodule_type].append(int(feat_idx))
    

    def _ablate_features(submodule_type, layer, feature_list):
        submodule_name = submodule_type_to_name(submodule_type).format(layer)
        if layer == -1:
            submodule = model.gpt_neox.embed_in
        else:
            submodule = load_submodule(model, submodule_name)

        # if we have sequence positions
        if len(feature_list) > 0 and isinstance(feature_list[0], tuple):
            seqpos_list, temp_feature_list = [], []
            for idx in range(len(feature_list)):
                seqpos_list.append(feature_list[idx][0])
                temp_feature_list.append(feature_list[idx][1])
            feature_list = temp_feature_list
        else:
            seqpos_list = None

        # if there are no features to ablate, just return the submodule output
        # if len(feature_list) == 0 and return_submodules is not None and submodule_name in return_submodules:
        #     saved_submodules[submodule_name] = submodule.output.save()
        #     return

        # load autoencoder
        autoencoder = dictionaries[submodule]

        if patch_type == "zero":
            patch_vector = t.zeros(dictionary_size).to("cuda:0")
        elif patch_type == "mean":
            patch_vector = mean_vectors[submodule]

        # encode activations into features
        is_resid = (type(submodule.output.shape) == tuple)
        if is_resid:
            x_clean = submodule.output[0]
        else:
            x_clean = submodule.output
        f_clean = autoencoder.encode(x_clean)
        x_hat_clean = autoencoder.decode(f_clean)
        example_error = x_clean - x_hat_clean

        if inverse:     # decode(f) yields x_hat_C
            # x_bar = mean_vectors[submodule]
            # x_hat_bar = autoencoder.decode(patch_vector)
            # mean_error = x_bar - x_hat_bar

            patch_copy = patch_vector.clone().repeat(f_clean.shape[0], f_clean.shape[1], 1)
            if seqpos_list is not None:
                patch_copy[:, seqpos_list, feature_list] = f_clean[:, seqpos_list, feature_list]
            else:
                patch_copy[:, :, feature_list] = f_clean[:, :, feature_list]
            f_C = patch_copy               # ablate f everywhere except feature indices

            saved_submodules[submodule].append(copy(feature_list))
            saved_submodules[submodule].append(copy(f_C))
            saved_submodules[submodule].append(f_clean.save())

            x_hat_C = autoencoder.decode(f_C)
            saved_submodules[submodule].append(copy(x_hat_C))
            saved_submodules[submodule].append(copy(autoencoder.decode(patch_vector)))
            # x_hat = x_hat_C + mean_error
            x_hat = x_hat_C

            if isinstance(include_errors[submodule], bool) and include_errors[submodule]:
                x_hat = x_hat_C + example_error
            elif include_errors[submodule].shape[0] > 1 and t.any(include_errors[submodule]).item():
                error_idx = include_errors[submodule].squeeze(1).nonzero().flatten()
                x_hat[:, error_idx, :] += example_error[:, error_idx, :]

        else:           # decode(f) yields x_hat_noC
            f_noC = f_clean.clone().detach()
            if seqpos_list is not None:
                f_noC[:, seqpos_list, feature_list] = patch_vector[feature_list]
            else:
                f_noC[:, :, feature_list] = patch_vector[feature_list]   # ablate f at feature indices
            
            x_hat_noC = autoencoder.decode(f_noC)
            x_hat = x_hat_noC
            # x_hat = x_hat_noC + example_error

            if isinstance(include_errors[submodule], bool) and include_errors[submodule]:
                pass
            elif include_errors[submodule].shape[0] > 1 and t.any(include_errors[submodule] == 0).item():
                error_idx = (include_errors[submodule] == 0).squeeze(1).nonzero().flatten()
                x_hat[:, error_idx, :] += example_error[:, error_idx, :]
            else:
                x_hat = x_hat_noC + example_error
        
        # replace submodule w/ autoencoder out
        if is_resid:
            submodule.output = (x_hat, *submodule.output[1:])
        else:
            submodule.output = x_hat

        # replace activations of submodule
        if return_submodules is not None and submodule_name in return_submodules:
            saved_submodules[submodule_name] = submodule.output.save()

    with model.trace(example):
        # from lowest layer to highest (does this matter in nnsight?)
        for layer in sorted(features_per_layer):
            for submodule_type in features_per_layer[layer]:
                _ablate_features(submodule_type, layer, features_per_layer[layer][submodule_type])
        if return_submodules is not None:
            for submodule_name in return_submodules:
                if submodule_name not in saved_submodules.keys():
                    submodule_type = "mlp" if "mlp" in submodule_name else "attn" if "attention" in submodule_name else "resid"
                    submodule_parts = submodule_name.split(".")
                    for idx, part in enumerate(submodule_parts):
                        if part.startswith("layer"):
                            layer = int(submodule_parts[idx+1])
                            break
                    _ablate_features(submodule_type, layer, [])
        model_out = model.embed_out.output.save()
        
    saved_submodules["model"] = model_out.value
    # for submodule in saved_submodules:
    #     if len(saved_submodules[submodule]) < 5:
    #         continue
    #     print(submodule)
    #     feats, f_C, f_clean, x_hat_C, x_hat_bar = saved_submodules[submodule]
    #     print(f_clean.value.nonzero())
    #     print(feats)# , f_clean.value[:, :, feats])
    #     print(f_C.nonzero().flatten())
    #     print()

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