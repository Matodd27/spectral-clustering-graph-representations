import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from scipy import sparse

def knn_graph(X, k=10, kernel='gaussian', symmetrise=True):
    Xtree = KDTree(X)
    knn_dist, knn_ind = Xtree.query(X, k=k)
    
    n = knn_ind.shape[0]
    k = np.minimum(knn_ind.shape[1],k)
    knn_ind = knn_ind[:,:k]
    knn_dist = knn_dist[:,:k]
    
    D = knn_dist*knn_dist
    eps = D[:,k-1]
    if kernel == 'gaussian':
        weights = np.exp(-4*D/eps[:,None])
    
    #Flatten knn data and weights
    knn_ind = knn_ind.flatten()
    weights = weights.flatten()

    #Self indices
    self_ind = np.ones((n,k))*np.arange(n)[:,None]
    self_ind = self_ind.flatten()

    #Construct sparse matrix and convert to Compressed Sparse Row (CSR) format
    W = sparse.coo_matrix((weights, (self_ind, knn_ind)),shape=(n,n)).tocsr()
    
    W.setdiag(0)
    W.eliminate_zeros()
    if symmetrise:
        return (W + W.transpose())/2
    return W


def fully_connected(X, metric='euclidean'):
    from scipy.spatial.distance import cdist
    return cdist(X, X, metric=metric)

def epsilon_graph(X, eps, metric='euclidean'):
    from scipy.spatial.distance import cdist
    d = cdist(X, X, metric='euclidean')
    d[d > eps] = 0
    return d

def adaptive_neighbour_graph():
    pass

def compute_laplacian(W):
    pass

def can_row_weights_from_dists(d_i, k, eps=1e-12):
    d = np.asarray(d_i, dtype=float)
    idx = np.argsort(d)
    d_sorted = d[idx]

    if k + 1 > d_sorted.size:
        k = max(1, d_sorted.size - 1)

    d_k1 = d_sorted[k]
    d_k = d_sorted[:k]

    gaps = d_k1 - d_k
    denom = np.sum(gaps) + eps
    w_k = gaps / denom
    return idx[:k], w_k

def adaptive_neighbour_graph_can(X, k=10, symmetrise=True):
    X = np.asarray(X, dtype=float)
    N = X.shape[0]

    XX = np.sum(X**2, axis=1, keepdims=True)
    dists = XX + XX.T - 2 * X @ X.T
    np.fill_diagonal(dists, np.inf)

    rows = []
    cols = []
    data = []

    for i in range(N):
        neigh_idx = np.where(np.isfinite(dists[i]))[0]
        d_i = dists[i, neigh_idx]

        local_idx, local_w = can_row_weights_from_dists(d_i, k=k)

        c = neigh_idx[local_idx]    
        rows.append(np.full(c.shape[0], i, dtype=np.int32))
        cols.append(c.astype(np.int32))
        data.append(local_w.astype(float))

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.concatenate(data)

    S = sparse.csr_matrix((data, (rows, cols)), shape=(N, N))

    if symmetrise:
        S = 0.5 * (S + S.T)

        row_sums = np.asarray(S.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1.0
        S = S.multiply(1.0 / row_sums[:, None])

    return S

def solve_adaptive_neighbour_row():
    pass