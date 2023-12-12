import torch as t
import json
import argparse
import pickle
import numpy as np
import pandas as pd
from itertools import product
from collections import defaultdict
from tqdm import tqdm
from datasets import load_dataset
from nnsight import LanguageModel
from causal_search import load_submodule
from dictionary_learning.dictionary import AutoEncoder
from dictionary_learning.buffer import ActivationBuffer

def load_autoencoder(submodule, autoencoder_path):
    submodule_width = submodule.out_features
    autoencoder_size = 8192
    if "_sz" in autoencoder_path:
        autoencoder_size = int(autoencoder_path.split("_sz")[1].split("_")[0].split(".")[0])
    elif "_dict" in autoencoder_path:
        autoencoder_size = int(autoencoder_path.split("_dict")[1].split("_")[0].split(".")[0])
    elif "/ae.pt" in autoencoder_path:
        autoencoder_size = int(autoencoder_path.split("/")[-2].split("_")[1])
    autoencoder = AutoEncoder(submodule_width, autoencoder_size).to("cuda")

    try:
        autoencoder.load_state_dict(t.load(autoencoder_path))
    except TypeError:
        autoencoder.load_state_dict(t.load(autoencoder_path).state_dict())

    return autoencoder

def cossim(a, b):
    """
    Compute the pairwise cosine similarities between the rows of a and b.
    """
    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)
    return t.mm(a, b.T)

def meanmax(m, max_dim=-1):
    return m.max(dim=max_dim).values.mean()

def mmcs(d1, d2, encoder=True):
    """
    Given two dictionaries d1, d2, compute the mean (over features in d1)
    maximum cossine similarity with a feature in d2.
    If encoder is True, then use encoder weights, else use decoder weights.
    """
    if encoder:
        return meanmax(cossim(d1.encoder.weight, d2.encoder.weight))
    else:
        return meanmax(cossim(d1.decoder.weight.T, d2.decoder.weight.T))

def neuron_similarity(autoencoders, models, submodules, dataset, on_weights=False):
    representations = defaultdict(list)
    correlations = defaultdict(lambda: defaultdict())
    similarities = defaultdict(lambda: defaultdict())
    pairs = defaultdict(lambda: defaultdict())

    for ae_id, (model_name, submodule_str, autoencoder_path) in tqdm(enumerate(zip(models, submodules, autoencoders)),
                                                                     desc="Model", total=len(autoencoders)):
        model = LanguageModel(model_name, device_map='cuda:0')
        submodule = load_submodule(model, submodule_str)
        autoencoder = load_autoencoder(submodule, autoencoder_path)

        if on_weights:
            # TODO: change back to encoder
            representations[ae_id] = autoencoder.decoder.weight.T
        else:
            print("Encoding dataset...")
            for ex_id, example in tqdm(enumerate(dataset), desc="Encoding", leave=False, total=len(dataset)):
                # print(example)
                with model.invoke(example) as invoker:
                    x = submodule.output
                    f = autoencoder.encode(x)
                    f_saved = f.save()        # [Batch size, seq len, dict size]
                representations[ae_id].append(f_saved.value[:, -1, :].squeeze().to("cuda"))
            representations[ae_id] = t.stack(representations[ae_id])
    
    print("Analyzing correlations...")
    for ae1, ae2 in tqdm(product(representations.keys(),
                                 representations.keys()),
                                 desc="Correlation",
                                 total=len(representations.keys())**2):
        if ae1 == ae2:                  # if same
            pass
        if ae2 in correlations[ae1]:    # if seen
            continue

        representations_1 = representations[ae1]
        representations_2 = representations[ae2]
        num_neurons_1 = representations_1.shape[-1]
        num_neurons_2 = representations_2.shape[-1]
        means_1 = t.mean(representations_1, dim=0, keepdim=True)
        means_2 = t.mean(representations_2, dim=0, keepdim=True)
        stddevs_1 = t.std(representations_1, dim=0, keepdim=True)
        stddevs_2 = t.std(representations_2, dim=0, keepdim=True)

        # TODO: change these back when using encoder
        covariance = (t.matmul(representations_1, representations_2.T) / representations_1.shape[1]
                    - t.matmul(means_1, means_2.T))
        correlation = covariance / t.matmul(stddevs_1, stddevs_2.T)
        correlation = t.abs(correlation).detach().to("cpu").numpy()
        
        # TODO: RSA — correlation distance
        # correlations[ae1][ae2] = correlation.max(axis=1)
        # correlations[ae2][ae1] = correlation.max(axis=0)
        correlations[ae1][ae2] = np.nanmax(correlation, axis=1)
        correlations[ae2][ae1] = np.nanmax(correlation, axis=0)

        similarities[ae1][ae2] = np.nanmean(correlations[ae1][ae2])# .mean()
        similarities[ae2][ae1] = np.nanmean(correlations[ae2][ae1])# .mean()

        pairs[ae1][ae2] = correlation.argmax(axis=1)
        pairs[ae2][ae1] = correlation.argmax(axis=0)
    
    """
    neuron_sort = {} 
    neuron_notated_sort = {}
    for network in tqdm(representations.keys(), desc='annotation'):
        neuron_sort[network] = sorted(list(range(num_neurons_1)),
                                  key=(lambda i: correlations[network][network2][i] for network2 in correlations[network]),
                                  reverse=True)
        neuron_notated_sort[network] = [
            (
                neuron,
                {
                    network2 : (
                        correlations[network][network2][neuron], 
                        pairs[network][network2][neuron],
                    ) 
                    for network2 in correlations[network]
                }
            ) 
            for neuron in neuron_sort[network]
        ]
    """
    
    output = {
        "correlations": correlations,
        "pairs": pairs,
        "similarities": similarities,
        #"neuron_sort": neuron_sort,
        #"neuron_notated_sort": neuron_notated_sort
    }
    return output


def representation_similarity(autoencoders, models, submodules, dataset, on_weights=False):
    representations = defaultdict(list)
    similarities = defaultdict(lambda: defaultdict())
    
    for ae_id, (model_name, submodule_str, autoencoder_path) in tqdm(enumerate(zip(models, submodules, autoencoders)),
                                                                     desc="Model", total=len(autoencoders)):
        model = LanguageModel(model_name, device_map='cuda:0')
        submodule = load_submodule(model, submodule_str)
        autoencoder = load_autoencoder(submodule, autoencoder_path)

        if on_weights:
            # TODO: change back to encoder
            representations[ae_id] = autoencoder.encoder.weight.T
        else:
            print("Encoding dataset...")
            for ex_id, example in tqdm(enumerate(dataset), desc="Encoding", leave=False, total=len(dataset)):
                with model.invoke(example) as invoker:
                    x = submodule.output
                    f = autoencoder.encode(x)
                    f_saved = f.save()
                representations[ae_id].append(f_saved.value[:, -1, :].squeeze().to("cuda"))    # [Batch size, seq len, dict size]
            representations[ae_id] = t.stack(representations[ae_id])

    for ae1, ae2 in tqdm(product(representations.keys(),
                                 representations.keys()),
                                 desc="Correlation",
                                 total=len(representations.keys())**2):
        if ae1 == ae2:
            # similarities[ae1][ae2] = 1.0
            pass
        if ae2 in similarities[ae1]:    # if seen
            continue

        X = representations[ae1]
        Y = representations[ae2]

        # TODO: put these back when switched back to encoder
        XtX_F = t.norm(t.matmul(X.T, X), p='fro').item()
        YtY_F = t.norm(t.matmul(Y.T, Y), p='fro').item()
        YtX_F = t.norm(t.matmul(Y.T, X), p='fro').item()

        sim = (YtX_F ** 2) / (XtX_F * YtY_F)
        similarities[ae1][ae2] = sim
        similarities[ae2][ae1] = sim
    
    output = {
        "similarities": similarities
    }
    return output


def plot_heatmap(similarities, savepath, labels=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    sns.set(font_scale=1.25)

    plt.figure(figsize=(15, 12))

    sorted_similarities = dict(sorted(similarities.items()))
    for item in similarities:
        sorted_similarities[item] = dict(sorted(similarities[item].items()))
    df = pd.DataFrame.from_dict(sorted_similarities)

    if labels:
        xticklabels = labels.split(",")
        yticklabels = labels.split(",")
    else:
        # TODO: put this back
        # xticklabels = sorted_similarities.keys()
        # yticklabels = sorted_similarities.keys()
        pass
    sns.heatmap(df, annot=False, fmt=".2f", linewidth=.5, cmap=sns.cubehelix_palette(as_cmap=True),)
    #             xticklabels=xticklabels, yticklabels=yticklabels)
    plt.xticks([3, 9, 15], ["0_8192", "1_32768", "2_32768"])
    plt.yticks([3, 9, 15], ["0_8192", "1_32768", "2_32768"])
    plt.hlines([6,12], 0, 18, colors=["black","black"])
    plt.vlines([6,12], 0, 18, colors=["black","black"])
    plt.title("CKA Similarity (weights)")
    plt.xlabel("Layer")
    plt.ylabel("Layer")
    plt.yticks(rotation=0)
    # plt.xticks(rotation=40, ha='right')
    plt.savefig(savepath, format="png", bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--autoencoders", "-a", type=str,
                        default='autoencoders/ae_mlp3_c4_lr0.0001_resample25000_dict32768.pt')
    # parser.add_argument("--dataset", "-d", type=str,
    #                     default="")
    parser.add_argument("--num_examples", "-n", type=int, default=100,
                        help="Number of example pairs to use in the causal search.")
    parser.add_argument("--submodules", "-s", type=str,
                        default="model.gpt_neox.layers.3.mlp.dense_4h_to_h")
    parser.add_argument("--models", "-m", type=str,
                        default="EleutherAI/pythia-70m-deduped")
    parser.add_argument("--plot_heatmap", type=str, default=None)
    parser.add_argument("--labels", type=str, default=None,
                        help="Labels for each autoencoder.")
    parser.add_argument("--representation_similarity", "-r", action="store_true",
                        help="Compute representation-level similarity instead of neuron-level recall.")
    parser.add_argument("--on_weights", action="store_true",
                        help="Whether to analyze weights (as opposed to activations on text samples.)")
    args = parser.parse_args()

    autoencoders, models, submodules = [], [], []
    autoencoders = args.autoencoders.split(",")
    n_autoencoders = len(autoencoders)
    if n_autoencoders > 1:
        if len(args.models.split(",")) == 1:
            models = [args.models] * n_autoencoders
        elif len(args.models.split(",")) == n_autoencoders:
            models = args.models.split(",")
        else:
            raise ValueError("Invalid number of models")
        
        if len(args.submodules.split(",")) == 1:
            submodules = [args.submodules] * n_autoencoders
        elif len(args.submodules.split(",")) == n_autoencoders:
            submodules = args.submodules.split(",")
        else:
            raise ValueError("Invalid number of submodules")
    
    text_stream = load_dataset("c4", "en", split="train", streaming=True)
    dataset = iter(text_stream)
    texts = []
    for _ in range(args.num_examples):
        texts.append(next(dataset)["text"])

    if args.representation_similarity:
        results = representation_similarity(autoencoders, models, submodules, texts, on_weights=args.on_weights)
    else:
        results = neuron_similarity(autoencoders, models, submodules, texts, on_weights=args.on_weights)
    # with open("linCKA_pythia-70m-deduped_100c4.pkl", "wb") as results_file:
    #     pickle.dump(dict(results["similarities"]), results_file)

    if args.plot_heatmap:
        plot_heatmap(results["similarities"], args.plot_heatmap, labels=args.labels)

    print(results)
