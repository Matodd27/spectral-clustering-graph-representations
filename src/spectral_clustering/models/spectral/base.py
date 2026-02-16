import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
from scipy.sparse.linalg import LinearOperator

class BaseSpectralClustering():
    def __init__(self, n_clusters: int, kind: str='symmetric'):
        self.n_clusters = n_clusters
        self.labels_ = None
        self.kind = kind

    def fit(self, W: np.ndarray, kind='symmetric', extra_dims=0):
        deg = None
        if isinstance(W, tuple):
            W, deg = W

        
        if kind == 'symmetric' or kind == 'rw':
            n = W.shape[0]
            is_op = isinstance(W, LinearOperator)

            ones = np.ones(n)
            if deg is None:
                d = np.asarray(W.sum(axis=1)).ravel() if sparse.isspmatrix(W) else np.asarray(W @ np.ones(n)).ravel()
            else:
                d = np.asarray(deg).ravel()
            d = np.maximum(d, 1e-12)

            dinv2 = 1.0 / np.sqrt(d)
            if is_op:
                def matvec_A(v):
                    v = np.asarray(v).ravel()
                    return dinv2 * (W @ (dinv2 * v))
                A = LinearOperator(shape=(n, n), matvec=matvec_A, dtype=float)
            else:
                Dinv2 = sparse.diags(dinv2, format="csr")
                A = Dinv2 @ W @ Dinv2


            # k smallest eigenpairs of Lsym
            k = self.n_clusters + extra_dims
            evalsA, evecs = sparse.linalg.eigsh(A, k=k, which="LA")
            evals = 1 - evalsA
            
            # Normalise rows
            if kind == 'symmetric':
                norms = np.sum(evecs*evecs, axis=1)
                row_scale = (norms + 1e-12) ** (-0.5)
                evecs = evecs * row_scale[:, None]
            elif kind == 'rw':
                evecs = dinv2[:, None] * evecs
        
        kmeans = KMeans(n_clusters=self.n_clusters).fit(evecs)
        self.labels_ = kmeans.labels_
        
    
    def fit_predict(self, W, kind='symmetric', extra_dims=0):
        self.fit(W, kind=kind, extra_dims=extra_dims)
        return self.labels_