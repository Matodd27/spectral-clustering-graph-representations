import numpy as np

def gauss_similarity(dists, sigma):
    W = np.exp(-dists**2 / (2 * sigma**2))
    W[~np.isfinite(dists)] = 0.0     
    np.fill_diagonal(W, 0.0)
    return W

def gauss_knn_similarity(dists):
    finite = np.isfinite(dists)
    d_tmp = np.where(finite, dists, np.nan)
    max_row = np.nanmax(d_tmp, axis=1, keepdims=True)

    max_row[max_row == 0] = 1e-12

    W = np.exp(-4 * dists**2 / (max_row**2))
    W[~finite] = 0.0
    np.fill_diagonal(W, 0.0)
    return W
