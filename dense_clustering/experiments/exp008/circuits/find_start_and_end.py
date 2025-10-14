"""
This will find the first cluster with max_size and the last cluster with
min_size elements.
"""

import numpy as np
import os
from collections import defaultdict
import pickle

max_size = 70
min_size = 7

if __name__ == '__main__':
    with open("/om/user/ericjm/results/dictionary-circuits/dense_clustering/exp008/idxs.pkl", "rb") as f:
        idxs = pickle.load(f)

    with open(os.path.join("/om/user/ericjm/results/dictionary-circuits/dense_clustering/exp008/clusters-pythia-70m-deduped-100k-kmeans-30k-dim.pkl"), "rb") as f:
        clusters = pickle.load(f)

    clusters = clusters[4000]
    cluster_is = defaultdict(list)
    for i, c in enumerate(clusters):
        cluster_is[c].append(i)
    # create a new cluster label (indexing) scheme so that the largest cluster is cluster 0
    # and the second largest is cluster 1, etc.
    new_i_to_old_i = np.argsort([len(cluster_is[c]) for c in range(4000)])[::-1]

    for i in range(4000):
        if len(cluster_is[new_i_to_old_i[i]]) <= max_size:
            print(f"First max_size cluster index: {i}")
            break
    for i in range(3999, -1, -1):
        if len(cluster_is[new_i_to_old_i[i]]) >= min_size:
            print(f"Last min_size cluster index: {i}")
            break
    # import code; code.interact(local=locals())

# First max_size cluster index: 408
# Last min_size cluster index: 1995
