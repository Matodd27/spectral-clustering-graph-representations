import numpy as np

def gauss_similarity(dists, sigma):
    return np.exp(-dists**2 / (2*sigma**2))

def gauss_knn_similarity(dists):
    return np.exp(-4 * dists**2/ (np.max(dists, axis=1)))