# import numpy as np
# from sklearn.cluster import KMeans
# from scipy.spatial.distance import cdist 
# from spectral_clustering.graphs import adaptive_neighbour_graph, compute_laplacian
# import cvxpy as cp

# class OneStepSpectral():
#     def __init__(self, n_clusters: int, kind: str='symmetric'):
#         self.n_clusters = n_clusters
#         self.labels_ = None
#         self.embedding_ = None
#         self.kind = kind
#         self.S = None
        
#     def update_F(self, S):
#         L = compute_laplacian(S, kind=self.kind)
#         evals, evecs = np.linalg.eigh(L)
#         return evecs[:, :self.n_clusters]
    
#     def update_S(self, X, F, gamma, lambda_):
#         N = X.shape[0]
#         S = np.zeros(shape=(N,N))
#         dists = cdist(X, X, metric='sqeuclidean')
#         f = cdist(F, F, metric='sqeuclidean')
        
#         d = dists + lambda_*f
        
#         for i in range(N):
#             s_i = solve_adaptive_neighbour_row(d[i], gamma)
#             S[i, :] = s_i
            
#         return S
            
#     def fit(self, X: np.ndarray, gamma, lambda_):
#         S = adaptive_neighbour_graph(X, gamma)
#         for i in range(10):
#             F = self.update_F(S)
#             S = self.update_S(X, F, gamma, lambda_)
            
#         F_norm = F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-12)

#         km = KMeans(n_clusters=self.n_clusters, n_init='auto', random_state=0)
#         self.labels_ = km.fit_predict(F_norm)
        
#         self.S = S
#         return self
    
#     def fit_predict(self, X, gamma, lambda_):
#         self.fit(X, gamma, lambda_)
#         return self.labels_