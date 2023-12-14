import zstandard as zstd
import json
import os
import io
import argparse
from tqdm import tqdm
from nnsight import LanguageModel
from dictionary_learning.buffer import ActivationBuffer
from dictionary_learning.dictionary import AutoEncoder
from dictionary_learning.training import trainSAE
from datasets import load_dataset
import torch as t


def randomize_except_embeddings(model, seed=14):
    def _reinitialize_weights(module, initializer_range=0.02):
        """Initialize the weights."""
        # Set random seed for reproducible randomized weights
        t.manual_seed(seed)

        if isinstance(module, (t.nn.Linear,)):
            # Slightly different from Mesh Transformer JAX which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, t.nn.Embedding):
            pass
        elif isinstance(module, t.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    for module in model.local_model.modules():
        _reinitialize_weights(module)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-70m-deduped", help="Model name.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--steps", type=int, default=10000, help="Number of training steps.")
    parser.add_argument("--resample_steps", type=int, default=1000,
                        help="Number of steps between resampling dead neurons.")
    parser.add_argument("--in_bsz", type=int, default=64, help="In batch size.")
    parser.add_argument("--contexts_per_step", type=int, default=10,
                        help="Number of contexts to train on per step.")
    parser.add_argument("--max_contexts_len", type=int, default=128,
                        help="Maximum length of context.")
    parser.add_argument("--layer_num", type=int, default=3,
                        help="Layer of the model at which to train the autoencoder.")
    parser.add_argument("--save_steps", type=int, default=None, help="Number of steps between saves.")
    parser.add_argument("--dict_size", type=int, default=2048, help="Number of features in learned dictionary.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Model device.")
    parser.add_argument("--seed", type=int, default=14, help="Random seed.")
    args = parser.parse_args()

    model = LanguageModel(args.model, device_map='cuda:0')
    # submodule = model.gpt_neox.layers[args.layer_num].mlp.dense_4h_to_h
    submodule = model.gpt_neox.layers[args.layer_num]
    # activation_dim = submodule.out_features
    t.manual_seed(args.seed)

    model_dir = args.model.split("/")[-1]
    submodule_dir = f"resid_out_layer_{args.layer_num}"
    save_dir = os.path.join("autoencoders", model_dir, submodule_dir, f"0_{args.dict_size}")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/config.json", "w") as config_file:
        config = {"activation_dim": 512, "dictionary_size": args.dict_size,
                  "entropy": False, "io": "out", "sparsity_penalty": 3e-3, "lr": args.lr,
                  "steps": args.steps, "layer": args.layer_num, "model": args.model,
                  "submodule": "resid_out", "resample_steps": args.resample_steps,
                  "warmup_steps": 10000}
        config_file.write(json.dumps(config))
    
    # random ablation
    # model = randomize_except_embeddings(model, seed=args.seed)

    text_stream = load_dataset("c4", "en", split="train", streaming=True)
    data = iter(text_stream)

    buffer = ActivationBuffer(
        data,
        model,
        submodule,
        in_batch_size = args.in_bsz,
        out_batch_size = args.max_contexts_len * args.contexts_per_step,
        out_feats = 512,
        n_ctxs = args.contexts_per_step * 100,
        is_hf=True,
    )

    ae = trainSAE(
        buffer,
        # activation_dim = activation_dim,
        activation_dim = 512,
        dictionary_size = args.dict_size,
        steps = args.steps,
        warmup_steps = 10000,
        lr = args.lr,
        sparsity_penalty = 3e-3,
        entropy=False,
        resample_steps = args.resample_steps,
        log_steps = args.resample_steps,
        save_steps = args.save_steps,
        save_dir = save_dir,
        device='cuda:0',
        seed=args.seed
    )

    t.save(ae.state_dict(), f"{save_dir}/ae.pt")
