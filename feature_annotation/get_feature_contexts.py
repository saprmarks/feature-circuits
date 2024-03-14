#%%
# Imports
import sys
# sys.path.insert(0, '../')
sys.path.append('/home/can/dictionary-circuits')

from nnsight import LanguageModel
from load_features import FeatureID, randomly_sample_feature_id, load_feature_list_from_file_rc, load_feature_list_from_file_bib
from dictionary_learning import ActivationBuffer
from dictionary_learning.interp import examine_dimension
from dictionary_learning.utils import zst_to_generator
from loading_utils import load_submodules_and_dictionaries
from circuitsvis.activations import text_neuron_activations
import json
from tqdm import tqdm
import numpy as np


# EXAMPLE
# feature_list = [
#     FeatureID('mlp', 0, 27101, '5_32768'),
#     FeatureID('resid', 0, 13833, '5_32768'),
#     FeatureID('mlp', 4, 10980, '5_32768'),
#     FeatureID('attn', 5, 32515, '5_32768'),
#     FeatureID('resid', 5, 1593, '5_32768'),
# ]

# Define parameters
SET_NAME = "sparse_rc"
TRAINING_RUN_NAME = "10_32768"

# Load feature lists
feature_list_sparse_RC = load_feature_list_from_file_rc(SET_NAME, f'/home/can/dictionary-circuits/feature_annotation/sets/sparse_RC_set.txt', TRAINING_RUN_NAME)
feature_list_dense_RC = load_feature_list_from_file_rc(SET_NAME, f'/home/can/dictionary-circuits/feature_annotation/sets/dense_RC_set.txt', TRAINING_RUN_NAME)
feature_list_sparse_bib = load_feature_list_from_file_bib(SET_NAME, f'/home/can/dictionary-circuits/feature_annotation/sets/sparse_BiB_set.txt', TRAINING_RUN_NAME)
feature_list_dense_bib = load_feature_list_from_file_bib(SET_NAME, f'/home/can/dictionary-circuits/feature_annotation/sets/dense_BiB_set.txt', TRAINING_RUN_NAME)
all_features_in_circuits = feature_list_sparse_RC + feature_list_dense_RC + feature_list_sparse_bib + feature_list_dense_bib

feature_list = feature_list_sparse_RC

N_SAMPLES = 25
SUBMODULE_TYPES = ['attn', 'mlp', 'resid']
N_LAYERS = 6
N_CTXS = 256
CTX_LEN = 128
SEED = 42
K = 10
device = 'cuda:0'
output_file_name = f'{SET_NAME}_{TRAINING_RUN_NAME}_contexts'


set_arg_1, set_arg_2= SET_NAME.split('_')
if set_arg_1 == 'sparse':
    USE_SPARSE_DICTIONARIES = True
    N_FEATURES = 512*64
elif set_arg_1 == 'dense':
    N_FEATURES = 512
    USE_SPARSE_DICTIONARIES = False
else:
    raise ValueError(f"Invalid set name: {SET_NAME}")

# Load the model and dictionaries
model = LanguageModel('EleutherAI/pythia-70m-deduped', device_map=device)
submodules, submodule_names, dictionaries = load_submodules_and_dictionaries(
        model,
        use_attn=True,
        use_mlp=True,
        use_resid=True,
        dict_path="/share/projects/dictionary_circuits/autoencoders/pythia-70m-deduped/",
        dict_size=512*64,
        dict_run_name=TRAINING_RUN_NAME,
        device=device,
)
submodule_names = {v: k for k, v in submodule_names.items()}
if not USE_SPARSE_DICTIONARIES:
    dictionaries = {k: None for k, v in dictionaries.items()}

#%%
# Setup Buffer
data = zst_to_generator('/share/data/datasets/pile/the-eye.eu/public/AI/pile/train/00.jsonl.zst')
buffer = ActivationBuffer(
    data,
    model,
    [submodules[0]], # doesn't matter which submodule, we only use the buffer for contexts
    out_feats=512,
    in_batch_size=128,
    n_ctxs=N_CTXS,
    ctx_len=CTX_LEN,
    device=device,
)
inputs = buffer.text_batch(batch_size=N_CTXS)

# Examine components
def eval_component(feature_id):
    feat_idx = feature_id.feature_idx
    submodule = submodule_names[feature_id.submodule_type + str(feature_id.layer_idx)]
    dictionary = dictionaries[submodule]
    ex_result = examine_dimension(
        model,
        submodule,
        inputs,
        dictionary,
        dim_idx=feat_idx,
        max_length=CTX_LEN,
        n_inputs=N_CTXS,
        k=K,
    )
    feat_dict = dict(
        set_name=feature_id.set_name,
        submodule_type=feature_id.submodule_type,
        layer_idx=feature_id.layer_idx,
        feature_idx=feature_id.feature_idx,
        training_run_name=feature_id.training_run_name,
    )
    return ex_result, feat_dict

def eval_circuit_components(results: dict, feature_list):
    for i, feat in tqdm(enumerate(feature_list), desc="Examining features", total=len(feature_list)):
        cur_result, feat_dict = eval_component(feat)
        results[i] = cur_result
        results[i]['component'] = feat_dict

def eval_random_components(results: dict):
    l = len(results)
    i = l
    while len(results) < N_SAMPLES:
        print(f'Completed {i-l} samples')
        feat = randomly_sample_feature_id(SET_NAME, SUBMODULE_TYPES, N_LAYERS, N_FEATURES, TRAINING_RUN_NAME, exclude_features=all_features_in_circuits)
        all_features_in_circuits.append(feat)
        cur_result, feat_dict = eval_component(feat)
        if np.any(np.array(cur_result['top_contexts'][-1][1]) > 1e-4): # Validate the lowest of top 10 contexts has some activations
            results[i] = cur_result
            results[i]['component'] = feat_dict
            i += 1

# Run and save output
results = dict()
if set_arg_2 == 'random':
    eval_random_components(results)
else:
    eval_circuit_components(results, feature_list)
with open(f'/home/can/dictionary-circuits/feature_annotation/contexts/{output_file_name}.json', 'w') as f:
    json.dump(results, f)
