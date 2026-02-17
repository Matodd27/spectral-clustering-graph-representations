import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from spectral_clustering.graphs.constructors import can_row_weights_from_dists, laplacian_sparse, row_normalise_csr

class PCAN:
    def __init__(
        self,
        n_clusters: int,
        k: int = 10,
        candidate_k: int | None = None,
        lambda_: float = 1.0,
        kind: str = "symmetric",
        symmetrise: bool = True,
        refresh_neighbours_every: int = 1,
        max_iter: int = 30,
        tol: float = 1e-3,
        random_state: int = 0,
        x_scale: str | float = "auto",
        n_jobs: int | None = None,
    ):
        self.n_clusters = n_clusters
        self.k = k
        self.candidate_k = candidate_k if candidate_k is not None else max(2 * k, k + 1)
        self.lambda_ = float(lambda_)
        self.kind = kind
        self.symmetrise = symmetrise
        self.refresh_neighbours_every = int(refresh_neighbours_every)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state
        self.x_scale = x_scale
        self.n_jobs = n_jobs

        self.labels_ = None
        self.embedding_ = None
        self.S_ = None

    def _auto_x_scale(self, X, sample_size=512, rng=None):
        rng = np.random.default_rng(0) if rng is None else rng
        n = X.shape[0]
        m = min(sample_size, n)
        idx = rng.choice(n, size=m, replace=False)
        Xs = X[idx]

        jdx = rng.choice(m, size=min(64, m), replace=False)
        diffs = Xs[:, None, :] - Xs[jdx][None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diffs, diffs)
        med = np.median(np.sqrt(np.maximum(d2, 0.0)))
        return 1.0 / np.maximum(med, 1e-12)

    def _compute_neighbours(self, Y, n_neighbors):
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean", n_jobs=self.n_jobs)
        nn.fit(Y)
        d, idx = nn.kneighbors(Y, return_distance=True)
        return idx[:, 1:], (d[:, 1:] ** 2)

    def _build_S_from_candidates(self, cand_idx, cand_d2) -> sparse.csr_matrix:
        N = cand_idx.shape[0]
        rows, cols, data = [], [], []

        for i in range(N):
            d_i = cand_d2[i]           
            local_idx, local_w = can_row_weights_from_dists(d_i, k=self.k)
            chosen = cand_idx[i, local_idx]

            rows.append(np.full(chosen.size, i, dtype=np.int32))
            cols.append(chosen.astype(np.int32))
            data.append(local_w.astype(float))

        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        data = np.concatenate(data)

        S = sparse.csr_matrix((data, (rows, cols)), shape=(N, N))

        if self.symmetrise:
            S = 0.5 * (S + S.T)

        S = row_normalise_csr(S)
        return S

    def _update_F(self, S):
        L = laplacian_sparse(S, kind=self.kind)
        vals, vecs = eigsh(L, k=self.n_clusters, which="SM")
        order = np.argsort(vals)
        return vecs[:, order]

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        N = X.shape[0]
        if self.k + 1 >= N:
            raise ValueError("Need k+1 < N.")
        if self.candidate_k < self.k + 1:
            raise ValueError("candidate_k must be >= k+1.")
        if self.candidate_k >= N:
            self.candidate_k = N - 1

        rng = np.random.default_rng(self.random_state)
        F = rng.normal(size=(N, self.n_clusters))
        F /= np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-12)

        if self.x_scale == "auto":
            alpha = self._auto_x_scale(X, rng=rng)
        else:
            alpha = float(self.x_scale)

        Y = np.hstack([alpha * X, np.sqrt(self.lambda_) * F])
        cand_idx, cand_d2 = self._compute_neighbours(Y, n_neighbors=self.candidate_k + 1)
        S = self._build_S_from_candidates(cand_idx, cand_d2)

        for t in range(self.max_iter):
            F_new = self._update_F(S)

            diff = np.linalg.norm(F_new - F, ord="fro") / np.maximum(np.linalg.norm(F, ord="fro"), 1e-12)
            F = F_new

            if (t + 1) % self.refresh_neighbours_every == 0:
                Y = np.hstack([alpha * X, np.sqrt(self.lambda_) * F])
                cand_idx, cand_d2 = self._compute_neighbours(Y, n_neighbors=self.candidate_k + 1)

            S = self._build_S_from_candidates(cand_idx, cand_d2)

            if diff < self.tol:
                break

        # final clustering
        F_norm = F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-12)
        km = KMeans(n_clusters=self.n_clusters, n_init="auto", random_state=self.random_state)
        self.labels_ = km.fit_predict(F_norm)

        self.embedding_ = F
        self.S_ = S
        return self

    def fit_predict(self, X: np.ndarray):
        return self.fit(X).labels_
