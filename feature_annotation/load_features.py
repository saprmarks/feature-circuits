import numpy as np
from collections import namedtuple

FeatureID = namedtuple('FeatureID', ['set_name', 'submodule_type', 'layer_idx', 'feature_idx', 'training_run_name'])

# Randomly sample FeatureIDs
def randomly_sample_feature_id(set_name, submodule_types, n_layers, n_features, training_run_name, exclude_features=[]):
    submodule_type = np.random.choice(submodule_types)
    layer_idx = np.random.choice(list(range(n_layers)))
    feature_idx = np.random.choice(list(range(n_features)))
    feature_id = FeatureID(set_name, submodule_type, int(layer_idx), int(feature_idx), training_run_name)
    if feature_id in exclude_features:
        return randomly_sample_feature_id(set_name, submodule_types, n_layers, n_features, training_run_name, exclude_features)
    return feature_id

#%%
# Load from ./sparse_RC_set.txt
def load_feature_list_from_file_rc(set_name, file_path):
    with open(file_path, 'r') as f:
        feature_list = []
        for line in f.readlines():
            if line == '\n':
                continue
            submod, feats = line.split(': ')
            submod_type, layer_idx = submod.split('_')
            layer_idx = int(layer_idx)
            feats = feats.strip("[|]|\n").split(', ')
            for feat in feats:
                if feat == '':
                    continue
                feature_list.append(FeatureID(set_name, submod_type, layer_idx, int(feat), 'RC'))
    return feature_list

def load_feature_list_from_file_bib(set_name, file_path):
    with open(file_path, 'r') as f:
        feature_list = []
        for line in f.readlines():
            submod, feat = line.strip("', \n").split('/')
            if feat == 'res':
                continue
            submod_type, layer_idx = submod.split('_')
            feat_idx = int(feat)
            layer_idx = int(layer_idx)
            feature_list.append(FeatureID(set_name, submod_type, layer_idx, feat_idx, 'bib'))
    return feature_list