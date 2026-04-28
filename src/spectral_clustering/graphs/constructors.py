import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from scipy import sparse

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import scipy.sparse as sp
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, aslinearoperator
import hnswlib

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
        return sp.csr_matrix((n, n))

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

    W = sp.coo_matrix((weights_f, (self_ind, knn_ind_f)), shape=(n, n)).tocsr()
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

def can_row_weights_from_dists(d_sorted, eps=1e-12):
    d_sorted = np.asarray(d_sorted, dtype=np.float64)
    d_k1 = d_sorted[-1]
    d_k = d_sorted[:-1]

    gaps = np.maximum(d_k1 - d_k, 0.0)
    denom = gaps.sum() + eps
    return gaps / denom

def adaptive_neighbour_graph_can(
    X,
    k=10,
    symmetrise=True,
    eps=1e-12,
    M=16,
    ef_construction=200,
    ef_search=100,
    num_threads=-1,
    dtype=np.float32,
):

    X = np.asarray(X, dtype=dtype, order="C")
    N, dim = X.shape

    if N <= 1:
        return sp.csr_matrix((N, N), dtype=dtype)

    k_eff = min(k, N - 1)
    q = min(k_eff + 2, N)

    # Build HNSW index
    index = hnswlib.Index(space="l2", dim=dim)
    index.init_index(max_elements=N, ef_construction=ef_construction, M=M)
    index.add_items(X, np.arange(N), num_threads=num_threads)
    index.set_ef(max(ef_search, q))

    # Query neighbours for all points
    nn_ids, nn_dists = index.knn_query(X, k=q, num_threads=num_threads)

    rows = np.repeat(np.arange(N, dtype=np.int32), k_eff)
    cols = np.empty(N * k_eff, dtype=np.int32)
    data = np.empty(N * k_eff, dtype=dtype)

    ptr = 0
    for i in range(N):
        ids_i = nn_ids[i]
        d_i = nn_dists[i]

        # Drop self if present
        keep = ids_i != i
        ids_i = ids_i[keep]
        d_i = d_i[keep]

        m = min(k_eff + 1, ids_i.size)
        if m <= 1:
            continue

        ids_i = ids_i[:m]
        d_i = d_i[:m]

        order = np.argsort(d_i)
        ids_i = ids_i[order]
        d_i = d_i[order]

        if ids_i.size < k_eff + 1:
            # Fall back if ANN returned too few usable neighbours
            kk = ids_i.size - 1
            if kk <= 0:
                continue
            w = can_row_weights_from_dists(d_i[:kk + 1], eps=eps)
            cols[ptr:ptr + kk] = ids_i[:kk]
            data[ptr:ptr + kk] = w.astype(dtype, copy=False)
            ptr += kk
        else:
            w = can_row_weights_from_dists(d_i[:k_eff + 1], eps=eps)
            cols[ptr:ptr + k_eff] = ids_i[:k_eff]
            data[ptr:ptr + k_eff] = w.astype(dtype, copy=False)
            ptr += k_eff

    rows = rows[:ptr]
    cols = cols[:ptr]
    data = data[:ptr]

    S = sparse.csr_matrix((data, (rows, cols)), shape=(N, N), dtype=dtype)

    if symmetrise:
        S = 0.5 * (S + S.T)

        row_sums = np.asarray(S.sum(axis=1)).ravel()
        inv = np.zeros_like(row_sums, dtype=dtype)
        mask = row_sums > 0
        inv[mask] = 1.0 / row_sums[mask]

        S = sparse.diags(inv) @ S

    return S

def laplacian_sparse(S: sparse.csr_matrix, kind: str = "symmetric") -> sparse.csr_matrix:
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


def hnsw_knn_connectivity_graph(
    X: np.ndarray,
    k: int,
    *,
    symmetrise: bool = True,
    metric: str = "l2",          # "l2", "cosine", or "ip"
    ef_construction: int = 200,
    M: int = 16,
    ef: int | None = None,
    num_threads: int = -1,
    dtype: np.dtype = np.float64,
):
    """
    Returns a sparse kNN graph with binary weights {0,1}, built using hnswlib.

    Important:
    - This returns a plain connectivity graph, NOT a Gaussian similarity graph.
    - If symmetrise=True, uses union symmetrisation: G <- max(G, G.T).
    """
    import hnswlib

    X = np.asarray(X, dtype=np.float32, order="C")
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")

    n, d = X.shape
    if not isinstance(k, int) or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}")
    if k >= n:
        raise ValueError(f"k must be < n; got k={k}, n={n}")

    index = hnswlib.Index(space=metric, dim=d)
    index.init_index(
        max_elements=n,
        ef_construction=ef_construction,
        M=M,
    )
    ids = np.arange(n)
    index.add_items(X, ids, num_threads=num_threads)

    # hnswlib recommends ef > k for query accuracy
    ef = max(k + 1, 2 * k if ef is None else ef)
    index.set_ef(ef)

    # Query k+1 because self is usually returned as a neighbour
    labels, _distances = index.knn_query(
        X,
        k=k + 1,
        num_threads=num_threads,
    )

    rows = []
    cols = []

    for i in range(n):
        nbrs = labels[i]
        nbrs = nbrs[nbrs != i]   # drop self
        nbrs = nbrs[:k]          # keep exactly k neighbours

        if nbrs.shape[0] != k:
            raise RuntimeError(
                f"After removing self, row {i} has only {nbrs.shape[0]} neighbours; expected {k}."
            )

        rows.extend([i] * k)
        cols.extend(nbrs.tolist())

    data = np.ones(len(rows), dtype=dtype)
    G = sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=dtype)

    if symmetrise:
        G = G.maximum(G.T)

    G.setdiag(0.0)
    G.eliminate_zeros()
    return G

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
            knn_graph_fn = hnsw_knn_connectivity_graph
        except NameError as e:
            raise NameError(
                "knn_graph is not defined. Pass knn_graph_fn=... or import knn_graph "
                "from spectral_clustering.graphs.constructors."
            ) from e

    K = knn_graph_fn(X, k=k, symmetrise=symmetrise)
    if not sp.isspmatrix_csr(K):
        K = K.tocsr()

    K = K.astype(dtype, copy=False)

    if zero_diagonal:
        K.setdiag(0.0)
        K.eliminate_zeros()

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
