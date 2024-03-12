
import os
from collections import defaultdict
import random
import pickle
import h5py

from tqdm.auto import tqdm
import numpy as np

import faiss

SAVE_DIR = "/om/user/ericjm/results/dictionary-circuits/dense_clustering/exp008"

# load up idxs
with open("/om/user/ericjm/results/dictionary-circuits/dense_clustering/exp008/idxs.pkl", "rb") as f:
    idxs = pickle.load(f)

# load up all the vectors from the hdf5 file
with h5py.File("/om/user/ericjm/results/dictionary-circuits/dense_clustering/exp008/gradients.h5", "r") as f:
    Gs = f["gradients"][:]

# plt.imshow(C)
# plt.xlim(0, 100)
# plt.ylim(100, 0)
# # check if matrix C has any NaNs
# np.isnan(C).any()
# np.isinf(C).any()

CLUSTER_COUNTS = [750, 1250, 2500, 4000]

results = dict()

random.seed(0)
np.random.seed(0)

for n_clusters in tqdm(CLUSTER_COUNTS):
    kmeans = faiss.Kmeans(
        d=30000,
        k=n_clusters,
        gpu=True,
        niter=100,
        spherical=True,
        verbose=True,
    )
    kmeans.train(Gs)
    distances, clusters_kmeans = kmeans.assign(Gs)
    results[n_clusters] = clusters_kmeans.tolist()

with open(os.path.join(SAVE_DIR, "clusters-pythia-70m-deduped-100k-kmeans-30k-dim.pkl"), "wb") as f:
    pickle.dump(results, f)

