#%%
# Setup
import zstandard as zstd
import io
import json
from nnsight import LanguageModel


model = LanguageModel('EleutherAI/pythia-70m-deduped', device_map='cuda:0')

# set up data as a generator
data_path = '/share/data/datasets/pile/the-eye.eu/public/AI/pile/train/00.jsonl.zst'
compressed_file = open(data_path, 'rb')
dctx = zstd.ZstdDecompressor()
reader = dctx.stream_reader(compressed_file)
text_stream = io.TextIOWrapper(reader, encoding='utf-8')

def generator():
    for line in text_stream:
        yield json.loads(line)['text']
data = generator()

def text_batch(batch_size):
        """
        Return a list of text
        """
        return [
            next(data) for _ in range(batch_size)
        ]
# %%
# Metric and 

# Tokenize sample
prompt_str = next(data)
prompt_tokid = model.tokenizer.encode(prompt_str)[:10]

# Metric is next-token prediction: Cross-Entropy Loss

# %%
# Cache gradients for a backwardpass
with model.invoke(prompt_tokid) as invoker:
     pass

# %%