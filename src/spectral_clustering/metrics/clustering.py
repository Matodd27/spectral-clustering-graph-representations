from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import numpy as np

def clustering_scores(labels_true, labels_pred):
    return {
        "ACC": clustering_accuracy(labels_true, labels_pred),
        "NMI": normalized_mutual_info_score(labels_true, labels_pred),
        "ARI": adjusted_rand_score(labels_true, labels_pred),
    }

def clustering_accuracy(labels_true, labels_pred):   
    import scipy.optimize as optim
    
    # Map true labels to 0,...,k-1
    labels_true_copy = labels_true.copy()
    unique_classes = np.unique(labels_true_copy)
    num_classes = len(unique_classes)
    
    ind = []
    for c in unique_classes:
        ind_c = labels_true_copy == c
        ind.append(ind_c)
        
    for i in range(num_classes):
        labels_true_copy[ind[i]] = i
        
    # Create cost matrix C and find minimum assignment of label permutations
    C = np.zeros((num_classes, num_classes), dtype=float)
    for i in range(num_classes):
        for j in range(num_classes):
            C[i][j] = np.sum((labels_pred == i) & (labels_true_copy != j))
    row_ind, col_ind = optim.linear_sum_assignment(C)
    
    return 100*(1-C[row_ind, col_ind].sum()/len(labels_pred))

def silhoutte_score(W, labels_):
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


# Optional convenience alias with correct spelling
silhouette_score = silhoutte_score
