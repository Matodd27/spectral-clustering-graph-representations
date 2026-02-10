from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import numpy as np

def clustering_scores(labels_true, labels_pred):
    return {
        "NMI": normalized_mutual_info_score(labels_true, labels_pred),
        "ARI": adjusted_rand_score(labels_true, labels_pred),
    }

def silhouette_score(W: np.ndarray, labels: np.ndarray) -> dict[int, float]:
    n = labels.shape[0]

    k = int(labels.max()) + 1
    if k <= 1:
        return {0: 0.0} if n > 0 else {}

    # Membership matrix M: (n, k)
    M = np.zeros((n, k), dtype=float)
    M[np.arange(n), labels] = 1.0
    sizes = M.sum(axis=0)  # (k,)

    # sums[i, c] = sum_{j in cluster c} W[i, j]
    sums = W @ M                    # (n, k)
    means = sums / sizes            # (n, k), broadcast

    diag = np.diag(W)

    # a(i): within-cluster mean similarity excluding self
    own_sizes = sizes[labels]
    denom_a = own_sizes - 1.0
    num_a = sums[np.arange(n), labels] - diag

    a = np.zeros(n, dtype=float)
    non_singleton = denom_a > 0
    a[non_singleton] = num_a[non_singleton] / denom_a[non_singleton]

    # b(i): maximum mean similarity to any other cluster
    means_other = means.copy()
    means_other[np.arange(n), labels] = -np.inf
    b = means_other.max(axis=1)

    # s(i)
    denom = np.maximum(a, b)
    s = np.zeros(n, dtype=float)
    valid = denom > 0
    s[valid] = (a[valid] - b[valid]) / denom[valid]

    # Singleton clusters -> 0 by convention
    s[~non_singleton] = 0.0

    # Per-cluster averages
    counts = np.bincount(labels, minlength=k).astype(float)
    sums_s = np.bincount(labels, weights=s, minlength=k)
    with np.errstate(divide="ignore", invalid="ignore"):
        avg_s = np.where(counts > 0, sums_s / counts, 0.0)

    return {c: float(avg_s[c]) for c in range(k) if counts[c] > 0}