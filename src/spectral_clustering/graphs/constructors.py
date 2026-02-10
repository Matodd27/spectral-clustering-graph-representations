import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from scipy import sparse
import cvxpy as cp

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
    return W


def fully_connected(X, metric='euclidean'):
    from scipy.spatial.distance import cdist
    return cdist(X, X, metric=metric)

def epsilon_graph(X, eps, metric='euclidean'):
    from scipy.spatial.distance import cdist
    d = cdist(X, X, metric='euclidean')
    d[d > eps] = 0
    return d

def solve_adaptive_neighbour_row(d_i, gamma):
    K = d_i.shape[0]

    s = cp.Variable(K)
    objective = cp.Minimize(d_i @ s + gamma * cp.sum_squares(s))
    constraints = [
        s >= 0,
        cp.sum(s) == 1
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP)

    return np.array(s.value, dtype=float)

def adaptive_neighbour_graph(X, gamma):    
    N = X.shape[0]
    S = np.zeros((N, N), dtype=float)
    
    XX = np.sum(X**2, axis=1, keepdims=True)
    dists = XX + XX.T - 2 * X @ X.T
    
    for i in range(N):
        dists[i, i] = np.inf

        neigh_idx = np.where(np.isfinite(dists[i]))[0]

        d_i = dists[i, neigh_idx]
        s_i = solve_adaptive_neighbour_row(d_i, gamma)

        S[i, neigh_idx] = s_i

    return S

def compute_laplacian(W):
    pass