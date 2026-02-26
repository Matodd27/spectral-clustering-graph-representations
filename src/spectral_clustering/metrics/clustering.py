import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

def clustering_scores(labels_partial, labels_pred):
    return {
        "ACC": clustering_accuracy(labels_partial, labels_pred),
        "NMI": normalized_mutual_info_score(labels_partial, labels_pred),
        "ARI": clustering_accuracy(labels_partial, labels_pred),
    }

def clustering_accuracy(labels_partial, labels_pred):   
    import scipy.optimize as optim
    
    # Map true labels to 0,...,k-1
    labels_partial_copy = labels_partial.copy()
    unique_classes = np.unique(labels_partial_copy)
    num_classes = len(unique_classes)
    
    ind = []
    for c in unique_classes:
        ind_c = labels_partial_copy == c
        ind.append(ind_c)
        
    for i in range(num_classes):
        labels_partial_copy[ind[i]] = i
        
    # Create cost matrix C and find minimum assignment of label permutations
    C = np.zeros((num_classes, num_classes), dtype=float)
    for i in range(num_classes):
        for j in range(num_classes):
            C[i][j] = np.sum((labels_pred == i) & (labels_partial_copy != j))
    row_ind, col_ind = optim.linear_sum_assignment(C)
    
    return 100*(1-C[row_ind, col_ind].sum()/len(labels_pred))

def silhouette_score(W, labels_):
    labels = np.asarray(labels_, dtype=np.int64)
    n = labels.shape[0]
    if n == 0:
        return {}

    k = int(labels.max()) + 1
    sizes = np.bincount(labels, minlength=k).astype(np.int64)

    # Helper to compute per-cluster means from sums, given denominators
    def _point_score_from_a_b(a, b):
        denom = np.maximum(a, b)
        s = np.zeros_like(a, dtype=float)
        mask = denom > 0
        s[mask] = (a[mask] - b[mask]) / denom[mask]
        return s

    # Detect sparse matrix without hard-requiring SciPy at import time
    is_sparse = False
    try:
        import scipy.sparse as sp
        is_sparse = sp.isspmatrix(W)
    except Exception:
        is_sparse = False

    s = np.zeros(n, dtype=float)

    if not is_sparse:
        # -------- Dense path --------
        W = np.asarray(W, dtype=float)
        if W.shape != (n, n):
            raise ValueError(f"Dense W must have shape {(n, n)}, got {W.shape}.")

        diag = np.diag(W)

        # Precompute reciprocal sizes for other-cluster means
        inv_sizes = 1.0 / np.maximum(sizes, 1)

        for i in range(n):
            # sums_c[c] = sum_j W[i, j] for j in cluster c  (O(n) in C via bincount)
            sums_c = np.bincount(labels, weights=W[i], minlength=k)

            ci = labels[i]

            # a(i): exclude self, average over all other points in own cluster
            denom_a = sizes[ci] - 1
            if denom_a > 0:
                a_i = (sums_c[ci] - diag[i]) / denom_a
            else:
                a_i = 0.0  # singleton cluster

            # b(i): maximum mean similarity to other clusters (full cluster mean)
            means_c = sums_c * inv_sizes
            means_c[ci] = -np.inf
            b_i = float(means_c.max())

            s[i] = _point_score_from_a_b(np.array([a_i]), np.array([b_i]))[0]

    else:
        # -------- Sparse path (observed-edges-only averaging) --------
        import scipy.sparse as sp
        W = W.tocsr()
        if W.shape != (n, n):
            raise ValueError(f"Sparse W must have shape {(n, n)}, got {W.shape}.")

        indptr, indices, data = W.indptr, W.indices, W.data

        for i in range(n):
            start, end = indptr[i], indptr[i + 1]
            cols = indices[start:end]
            vals = data[start:end]

            ci = labels[i]

            if cols.size == 0:
                # isolated node in the graph
                s[i] = 0.0
                continue

            lab_cols = labels[cols]

            # sums_c[c] = sum over observed edges from i to cluster c
            sums_c = np.bincount(lab_cols, weights=vals, minlength=k)
            # cnt_c[c] = number of observed edges from i to cluster c
            cnt_c = np.bincount(lab_cols, minlength=k)

            # If self-edge exists, remove it (rare if you zeroed diagonal, but safe)
            self_mask = (cols == i)
            if np.any(self_mask):
                self_w = float(vals[self_mask].sum())
                sums_c[ci] -= self_w
                cnt_c[ci] -= int(self_mask.sum())

            # a(i): mean over observed within-cluster edges
            if sizes[ci] <= 1 or cnt_c[ci] <= 0:
                a_i = 0.0
            else:
                a_i = sums_c[ci] / cnt_c[ci]

            # b(i): max mean over observed edges to other clusters
            # clusters with cnt==0 contribute mean 0 (no observed evidence of similarity)
            means_c = np.zeros(k, dtype=float)
            nonzero = cnt_c > 0
            means_c[nonzero] = sums_c[nonzero] / cnt_c[nonzero]
            means_c[ci] = -np.inf
            b_i = float(means_c.max())

            s[i] = _point_score_from_a_b(np.array([a_i]), np.array([b_i]))[0]

    # Per-cluster averages
    sum_s = np.bincount(labels, weights=s, minlength=k)
    out = {c: float(sum_s[c] / sizes[c]) for c in range(k) if sizes[c] > 0}
    return out

from scipy import stats

methods = ['knn', 'fc', 'adaptive', 'biclique', 'epsilon', 'PCAN']
labels = ['kNN', 'FC', 'Adaptive', 'Biclique', r'$\epsilon$-graph', 'PCAN']

def bar_comparison(bar1, bar2, methods=methods, labels=labels, legend=['No extra dimensions', 'Extra dimensions'], xlabel='Graph-building methodology', filename='results', ylim=(75, 100)):
    def summarise(df, cols):
        n = df.shape[0]
        mean = df[cols].mean()
        std = df[cols].std(ddof=1)
        sem = std / np.sqrt(n)
        tcrit = stats.t.ppf(0.975, df=n-1)
        ci = tcrit * sem
        return mean, ci

    base_mean, base_ci = summarise(bar1, methods)
    extra_mean, extra_ci = summarise(bar2, methods)

    if bar1[methods].max().max() <= 1.2:
        base_mean *= 100
        extra_mean *= 100
        base_ci *= 100
        extra_ci *= 100

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=300)

    ax.bar(x - width/2, base_mean, width,
        yerr=base_ci, capsize=3,
        label=legend[0],
        edgecolor="black", linewidth=0.8)

    ax.bar(x + width/2, extra_mean, width,
        yerr=extra_ci, capsize=3,
        label=legend[1],
        edgecolor="black", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean accuracy (%)")
    ax.set_xlabel(xlabel)
    ax.set_ylim(ylim)

    ax.grid(axis='y', linestyle='-', linewidth=0.5, alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.15),
        ncol=2,
        frameon=False
    )

    plt.tight_layout()
    plt.savefig(f"charts/{filename}.pdf", bbox_inches="tight")
    plt.savefig(f"charts/{filename}.png", bbox_inches="tight")
    plt.show()
    
def run_iters(X, Y, methods=['knn', 'fc', 'adaptive', 'biclique', 'pcan'], params=None, kind='symmetric', extra_dims=0, labels_partial=None, iters=100, num_clusters=10):
    from spectral_clustering.models.spectral import BaseSpectralClustering, PCAN
    from spectral_clustering.graphs.constructors import knn_graph, fully_connected, adaptive_neighbour_graph_can, compute_biclique_kr, epsilon_graph
    
    methods_to_fn = {
        'knn': lambda X: knn_graph(X, k=10),
        'fc': lambda X: fully_connected(X, symmetrise=True),
        'adaptive': lambda X: adaptive_neighbour_graph_can(X, k=10, symmetrise=True),
        'biclique': lambda X: compute_biclique_kr(X, k=10, r=4, symmetrise=True),
        'epsilon': lambda X: epsilon_graph(X, epsilon=0.5, symmetrise=True),
    }
    
    if not isinstance(X, tuple):
        X = (X,)
    if not isinstance(Y, tuple):
        Y = (Y,)
    if not isinstance(labels_partial, tuple):
        labels_partial = (labels_partial,)
        
    pcan_flag = False
    results = []
    spectral = BaseSpectralClustering(n_clusters=num_clusters, kind=kind)
    if 'pcan' in methods:
        pcan = PCAN(n_clusters=num_clusters, k=10, lambda_=params['lambda'], kind=kind, symmetrise=True)
    for x in range(len(X)):
        temp_df = pd.DataFrame(columns=methods)
        temp_df_extra = pd.DataFrame(columns=methods)
        Ws = [methods_to_fn[m](X[x]) for m in methods]
        for i in range(iters):
            for method_idx, method in enumerate(methods):
                W = Ws[method_idx]
                if extra_dims == 0:
                    if pcan_flag and method == 'pcan':
                        labels = pcan.fit_predict(X[x])
                    else:
                        labels = spectral.fit_predict(W, labels_true=labels_partial[x])
                    temp_df.loc[i, method] = clustering_accuracy(Y[x], labels)
                if extra_dims > 0:
                    max_acc = np.zeros(shape=(extra_dims,))
                    for ed in range(extra_dims):
                        if method == 'pcan':
                            labels = pcan.fit_predict(X[x])
                        else:
                            labels = spectral.fit_predict(W, extra_dims=ed, labels_true=labels_partial[x])
                        max_acc[ed] = clustering_accuracy(Y[x], labels)
                    temp_df.loc[i, method] = max_acc[0]
                    temp_df_extra.loc[i, method] = max_acc.max()
        if extra_dims > 0:
            results.append((temp_df, temp_df_extra))
        else:            
            results.append((temp_df))
    return results