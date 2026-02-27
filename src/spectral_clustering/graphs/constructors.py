import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from scipy import sparse

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, aslinearoperator

import numpy as np
from scipy import sparse
from sklearn.neighbors import KDTree

def knn_graph(X, k=10, kernel='gaussian', symmetrise=True, *, hnsw_M=16, hnsw_ef_construction=200, hnsw_ef=50):
    X = np.asarray(X)
    n, m = X.shape

    if k < 1:
        raise ValueError("k must be >= 1")

    if m <= 5:
        # Exact kNN
        k += 1
        Xtree = KDTree(X)
        knn_dist, knn_ind = Xtree.query(X, k=min(k, n))
    else:
        import hnswlib

        X32 = np.ascontiguousarray(X, dtype=np.float32)

        # Build index
        index = hnswlib.Index(space='l2', dim=m)
        index.init_index(max_elements=n, ef_construction=hnsw_ef_construction, M=hnsw_M)
        index.add_items(X32, np.arange(n))
        index.set_ef(hnsw_ef)

        kq = min(k + 1, n)
        labels, dists2 = index.knn_query(X32, k=kq)

        # Drop self neighbour if present
        if kq > 1:
            row_ids = np.arange(n)[:, None]
            self_mask = (labels == row_ids)

            if np.all(self_mask[:, 0]):
                knn_ind = labels[:, 1:kq]
                knn_dist = np.sqrt(dists2[:, 1:kq])
            else:
                knn_ind = np.empty((n, min(k, n-1)), dtype=int)
                knn_dist = np.empty((n, min(k, n-1)), dtype=float)
                for i in range(n):
                    keep = labels[i][labels[i] != i]
                    keep_d2 = dists2[i][labels[i] != i]
                    take = min(k, keep.shape[0])
                    knn_ind[i, :take] = keep[:take]
                    knn_dist[i, :take] = np.sqrt(keep_d2[:take])
        else:
            # n==1 case: no neighbours
            knn_ind = np.empty((n, 0), dtype=int)
            knn_dist = np.empty((n, 0), dtype=float)

    # Ensure correct shapes if k > n
    if knn_ind.ndim == 1:
        knn_ind = knn_ind[:, None]
        knn_dist = knn_dist[:, None]

    n = knn_ind.shape[0]
    k_eff = min(knn_ind.shape[1], k)
    knn_ind = knn_ind[:, :k_eff]
    knn_dist = knn_dist[:, :k_eff]

    if k_eff == 0:
        return sparse.csr_matrix((n, n))

    D = knn_dist * knn_dist 
    eps = D[:, k_eff - 1].copy()
    eps = np.maximum(eps, 1e-12)

    if kernel == 'gaussian':
        weights = np.exp(-4.0 * D / eps[:, None])
    else:
        raise ValueError(f"Unsupported kernel: {kernel}")

    knn_ind_f = knn_ind.reshape(-1)
    weights_f = weights.reshape(-1)

    self_ind = np.repeat(np.arange(n), k_eff)

    W = sparse.coo_matrix((weights_f, (self_ind, knn_ind_f)), shape=(n, n)).tocsr()
    W.setdiag(0)
    W.eliminate_zeros()

    if symmetrise:
        W = (W + W.transpose()) * 0.5

    return W

def fully_connected(X, metric='euclidean', kernel='gaussian'):
    from scipy.spatial.distance import cdist
    d = cdist(X, X, metric=metric)
    if kernel == 'gaussian':
        d = np.exp((-d*d)/np.quantile(d[d>0], 0.2)**2)
    return d
    


def epsilon_graph(X, eps, metric='euclidean', kernel='gaussian'):
    from scipy.spatial.distance import cdist
    d = cdist(X, X, metric='euclidean')
    d[d > eps] = 0
    if kernel == 'gaussian':
        d = np.exp((-d*d)/np.quantile(d[d>0], 0.2)**2)
    return d

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

def laplacian_sparse(S: sparse.csr_matrix, kind: str = "symmetric") -> sparse.csr_matrix:
    """
    Build Laplacian from sparse affinity.
    """
    S = S.tocsr()
    d = np.asarray(S.sum(axis=1)).ravel()

    if kind == "unnormalized":
        return sparse.diags(d, format="csr") - S

    if kind == "randomwalk":
        d_inv = 1.0 / np.maximum(d, 1e-12)
        Dinv = sparse.diags(d_inv, format="csr")
        I = sparse.identity(S.shape[0], format="csr")
        return I - Dinv @ S

    if kind == "symmetric":
        d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-12))
        Dinv2 = sparse.diags(d_inv_sqrt, format="csr")
        I = sparse.identity(S.shape[0], format="csr")
        return I - (Dinv2 @ S @ Dinv2)

    raise ValueError(f"Unknown Laplacian kind: {kind}")

def row_normalise_csr(S: sparse.csr_matrix) -> sparse.csr_matrix:
    row_sums = np.asarray(S.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    return S.multiply(1.0 / row_sums[:, None])

@dataclass(frozen=True)
class BicliqueKR:
    Kr: Union[LinearOperator, np.ndarray]
    degree: np.ndarray
    K_base: sp.csr_matrix
    delta: np.ndarray
    rho: float


def compute_biclique_kr(
    X: np.ndarray,
    k: int = 10,
    r: int = 2,
    return_operator: bool = True,
    symmetrise: bool = True,
    zero_diagonal: bool = False,
    include_scale_factor: bool = False,
    dtype: np.dtype = np.float64,
    knn_graph_fn=None,
):
    if X.ndim != 2:
        raise ValueError(f"X must be 2D with shape (n, d). Got shape {X.shape}.")

    n, d = X.shape
    if n < 2:
        raise ValueError("Need n >= 2 data points.")

    if not isinstance(k, int) or k <= 0:
        raise ValueError(f"k must be a positive int. Got {k}.")

    if not isinstance(r, int) or r < 2:
        raise ValueError(f"r must be an int >= 2. Got {r}.")

    if r % 2 != 0:
        raise ValueError(f"Saito biclique contraction assumes even r. Got r={r}.")

    if knn_graph_fn is None:
        try:
            knn_graph_fn = knn_graph
        except NameError as e:
            raise NameError(
                "knn_graph is not defined. Pass knn_graph_fn=... or import knn_graph "
                "from spectral_clustering.graphs.constructors."
            ) from e

    # 1) Base sparse Gram/affinity matrix K
    K = knn_graph_fn(X, k=k, symmetrise=symmetrise)
    if not sp.isspmatrix_csr(K):
        K = K.tocsr()

    K = K.astype(dtype, copy=False)

    if zero_diagonal:
        K.setdiag(0.0)
        K.eliminate_zeros()

    # 2) Precompute delta and rho from base K
    ones = np.ones(n, dtype=dtype)
    delta = np.asarray(K @ ones).ravel()     # delta_i = sum_j K_ij
    rho = float(delta.sum())                # rho = sum_{i,j} K_ij

    # Constants from the contraction form
    # K^{(r)}_ij = n^(r-2) * [ K_ij + alpha*(delta_i + delta_j) + beta*rho ]
    alpha = (r - 2) / (2.0 * n)
    beta = ((r - 2) ** 2) / (4.0 * (n ** 2))

    scale = (n ** (r - 2)) if include_scale_factor else 1.0

    # 3) Degree of K^{(r)} without forming K^{(r)}:
    degree = (delta + alpha * (delta * n + rho) + beta * rho * n) * scale
    degree = degree.astype(dtype, copy=False)

    if return_operator:
        def _matvec(v: np.ndarray) -> np.ndarray:
            v = np.asarray(v, dtype=dtype).reshape(-1)
            if v.shape[0] != n:
                raise ValueError(f"Vector length {v.shape[0]} does not match n={n}.")

            s1 = float(v.sum())              # 1^T v
            s2 = float(delta @ v)            # delta^T v

            out = np.asarray(K @ v).ravel()  # sparse matvec
            # Add rank-2 term: alpha*(delta*s1 + 1*s2)
            if alpha != 0.0:
                out = out + alpha * (delta * s1 + s2)
            # Add rank-1 constant term: beta*rho*1*s1
            if beta != 0.0 and rho != 0.0:
                out = out + (beta * rho * s1)

            if scale != 1.0:
                out = out * scale

            return out.astype(dtype, copy=False)

        Kr = LinearOperator(shape=(n, n), matvec=_matvec, dtype=dtype)

    else:
        Kr = K.toarray()
        if alpha != 0.0:
            Kr += alpha * (delta[:, None] + delta[None, :])
        if beta != 0.0 and rho != 0.0:
            Kr += (beta * rho)
        if scale != 1.0:
            Kr *= scale
        Kr = np.asarray(Kr, dtype=dtype)

    return BicliqueKR(Kr=Kr, degree=degree, K_base=K, delta=delta, rho=rho)
