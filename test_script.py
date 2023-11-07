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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--steps", type=int, default=10000, help="Number of training steps.")
    parser.add_argument("--resample_steps", type=int, default=1000,
                        help="Number of steps between resampling dead neurons.")
    parser.add_argument("--in_bsz", type=int, default=64, help="In batch size.")
    parser.add_argument("--contexts_per_step", type=int, default=10,
                        help="Number of contexts to train on per step.")
    parser.add_argument("--max_contexts_len", type=int, default=128,
                        help="Maximum length of context.")
    parser.add_argument("--save_steps", type=int, default=None, help="Number of steps between saves.")
    parser.add_argument("--dict_size", type=int, default=2048, help="Number of features in learned dictionary.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Model device.")
    args = parser.parse_args()

    model = LanguageModel('EleutherAI/pythia-70m-deduped', device_map='cuda:0')
    submodule = model.gpt_neox.layers[3].mlp.dense_4h_to_h

    text_stream = load_dataset("c4", "en", split="train", streaming=True)
    data = iter(text_stream)

    buffer = ActivationBuffer(
        data,
        model,
        submodule,
        in_batch_size = args.in_bsz,
        out_batch_size = args.max_contexts_len * args.contexts_per_step,
        n_ctxs = args.contexts_per_step * 50,
        is_hf=True,
        device='cuda:0'
    )

    ae = trainSAE(
        buffer,
        activation_dim=512,
        dictionary_size = args.dict_size,
        steps = args.steps,
        lr = args.lr,
        sparsity_penalty = 6e-3,
        entropy=False,
        resample_steps = args.resample_steps,
        log_steps = args.resample_steps,
        save_steps = args.save_steps,
        save_dir="autoencoders/",
        device='cuda:0'
    )

    t.save(ae, f"autoencoders/ae_mlp3_c4_lr{args.lr}_resample{args.resample_steps}_dict{args.dict_size}.pt")