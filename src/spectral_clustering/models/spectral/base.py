import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans

class BaseSpectralClustering():
    def __init__(self, n_clusters: int, kind: str='symmetric'):
        self.n_clusters = n_clusters
        self.labels_ = None
        self.kind = kind

    def fit(self, W: np.ndarray, kind='normalised'):
        if kind == 'normalised' or kind == 'rw':
            n = W.shape[1]
            d = W*np.ones(n)
            
            # L is symmetric and positive definite, hence SVD and EVD equivalent
            D = sparse.spdiags(d**(-1/2), 0, n, n)
            A = D*W*D
            
            u,s,vT = sparse.linalg.svds(A, k=self.n_clusters)
            
            evals = 1 - s
            ind = np.argsort(evals)
            evals = evals[ind]
            evecs = u[:, ind]
            
            # Normalise rows
            if kind == 'normalised':
                norms = np.sum(evecs*evecs, axis=1)
                T = sparse.spdiags(norms**(-1/2), 0, n, n)
                evecs = T@evecs    
            elif kind == 'rw':
                evecs = D@evecs
        
        kmeans = KMeans(n_clusters=self.n_clusters).fit(evecs)
        self.labels_ = kmeans.labels_
        
    
    def fit_predict(self, W):
        self.fit(W)
        return self.labels_