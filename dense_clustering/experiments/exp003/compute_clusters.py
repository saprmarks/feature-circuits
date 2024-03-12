
import os
from collections import defaultdict
import random
import pickle

from tqdm.auto import tqdm
import numpy as np
import torch
import sklearn.cluster

SAVE_DIR = "/om/user/ericjm/results/dictionary-circuits/dense_clustering/exp002"

idxs, C = torch.load(os.path.join(SAVE_DIR, "similarity.pt"))
C = C.cpu().numpy()
C = np.clip(C, -1, 1)
C = 1 - np.arccos(C) / np.pi

# plt.imshow(C)
# plt.xlim(0, 100)
# plt.ylim(100, 0)
# # check if matrix C has any NaNs
# np.isnan(C).any()
# np.isinf(C).any()

CLUSTER_COUNTS = [100, 300, 500, 1000]

results = dict()

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

for n_clusters in tqdm(CLUSTER_COUNTS):
    clusters_labels = sklearn.cluster.SpectralClustering(
        n_clusters=n_clusters, 
        affinity='precomputed', 
        n_init=30,
        random_state=0).fit_predict(C)
    results[n_clusters] = clusters_labels.tolist()

with open(os.path.join(SAVE_DIR, "clusters.pkl"), "wb") as f:
    pickle.dump(results, f)
