import numpy as np

def knn_graph(X, k=10, metric='euclidean', symmetrise=True):
    from scipy.spatial.distance import cdist
    
    d = cdist(X, X, metric=metric)
    
    n = d.shape[0]
    idx_knn = np.argpartition(d, kth=k, axis=1)[:, :k+1]
    mask = np.zeros_like(d, dtype=bool)
    rows = np.arange(n)[:, None]
    mask[rows, idx_knn] = True
    if symmetrise:
        mask = np.logical_or(mask, mask.T)
    d = d * mask
    np.fill_diagonal(d, 0)
    
    return d

def fully_connected(X, metric='euclidean'):
    from scipy.spatial.distance import cdist
    return cdist(X, X, metric=metric)

def epsilon_graph(X, eps, metric='euclidean'):
    from scipy.spatial.distance import cdist
    d = cdist(X, X, metric='euclidean')
    d[d > eps] = 0
    return d

def adaptive_neighbour_graph(X, gamma):
    import cvxpy as cp
    
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

def compute_laplacian(S, kind='symmetric'):
    d = S.sum(axis=1)
    n = S.shape[0]
    if kind == 'unnormalised':
        D = np.diag(d)
        return D - S
    elif kind == 'symmetric':
        d_inv_sqrt = 1.0 / np.sqrt(d)
        D_inv_sqrt = np.diag(d_inv_sqrt)
        
        return np.eye(n) - D_inv_sqrt @ S @ D_inv_sqrt
    elif kind == 'rw':
        d_inv = 1.0 / d
        D_inv = np.diag(d_inv)
        return np.eye(n) - D_inv @ S
    raise ValueError('Laplacian kind not defined, valid kinds are: unnormalised, symmetric, rw')