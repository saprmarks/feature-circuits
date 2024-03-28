
import random
import time
import os
import sys

import numpy as np
import torch
from tqdm import trange

from transformers import GPTNeoXForCausalLM, AutoTokenizer
from datasets import load_dataset


def tokenize_split(txt, tokenizer):
    """Splits string `txt` according to how `tokenizer` tokenizes it."""
    return tokenizer.batch_decode(tokenizer.encode(txt))


if __name__ == '__main__':
    
    output_dir = "/home/can/data/trajectory" 
    model_name = "pythia-70m-deduped"
    # model_name = model_names[int(sys.argv[1])]
    steps = [0] + [2**i for i in range(10)] + list(range(1000, 144000, 1000))
    device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
  
    k_idx = int(sys.argv[1]) # index 0...153 where there are 154 checkpoints
    step = steps[k_idx]

    if os.path.exists(os.path.join(output_dir, f"{model_name}-step{step}.npy")):
        print("Run already performed. Aborting...")
        sys.exit(1)

    # load the_pile test set
    dataset = load_dataset("json", data_files="/home/can/data/pile/test.jsonl.zst", cache_dir="/home/can/data/pile/") 
    dataset = dataset['train'] # this is the test set of the pile (just confusing hf thing)
    print("loaded dataset")
    print("now loading model")
    model = GPTNeoXForCausalLM.from_pretrained(
        f"EleutherAI/{model_name}",
        revision=f"step{step}",
        cache_dir=f"/home/can/feature_clustering/model_cache/{model_name}/step{step}",
    ).to(device)
    print("done loading model")
    
    tokenizer = AutoTokenizer.from_pretrained(
        f"EleutherAI/{model_name}",
        revision=f"step{step}",
        cache_dir=f"/home/can/feature_clustering/model_cache/{model_name}/step{step}",
    )
    print("done loading tokenizer")
    print("now evaluating model...") 
    results = []
    with torch.inference_mode():
        for i in trange(20000):
            prompt = dataset[i]['text']
            tokens = tokenizer(prompt, return_tensors='pt', max_length=1024, truncation=True).to(device)
            logits = model(**tokens).logits
            targets = tokens.input_ids
            ls = torch.nn.functional.cross_entropy(logits[0, :-1, :], targets[0, 1:], reduction='none')
            results.append(ls.tolist())
    total_length = sum(len(result) for result in results)

    results_arr = np.zeros(total_length, dtype=np.float32)
    j = 0
    for x in results:
        results_arr[j:j+len(x)] = np.array(x, dtype=np.float32)
        j += len(x)
    np.save(os.path.join(output_dir, f"{model_name}-step{step}.npy"), results_arr)


