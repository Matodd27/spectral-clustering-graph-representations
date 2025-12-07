import numpy as np

class BaseSpectralClustering():
    def __init__(self, n_clusters: int, kind: str='symmetric'):
        self.n_clusters = n_clusters
        self.labels_ = None
        self.embedding_ = None
        self.kind = kind

    def fit(self, S: np.ndarray):
        from spectral_clustering.graphs.constructors import compute_laplacian
        from scipy.sparse.linalg import eigsh
        from sklearn.cluster import KMeans
        
        L = compute_laplacian(S, kind=self.kind)
        
        evals, evecs = np.linalg.eigh(L)
        U = evecs[:, :self.n_clusters]
        
        U_norm = U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)

        km = KMeans(n_clusters=self.n_clusters, n_init='auto', random_state=0)
        self.labels_ = km.fit_predict(U_norm)
            
        return self
    
    def fit_predict(self, S):
        self.fit(S)
        return self.labels_