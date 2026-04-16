import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix
import scipy.sparse as sp
import time

plt.rcParams.update({
    # --- Font ---
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'axes.unicode_minus': False,

    # --- Font sizes ---
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,   # was 11
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,

    # --- Figure ---
    'figure.figsize': (7, 4.2),
    'figure.dpi': 300,
    'figure.facecolor': 'white',

    # --- Axes ---
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 0.8,
    'axes.grid': False,

    # --- Grid ---
    'grid.color': '0.85',
    'grid.linestyle': '-',
    'grid.linewidth': 0.5,

    # --- Ticks ---
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,

    # --- Lines ---
    'lines.linewidth': 1.5,

    # --- Legend ---
    'legend.frameon': False,

    # --- Savefig ---
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white'
})

COLORS = {
    'raw': '#666666',
    'vae': '#0072B2',
    'simclr': '#009E73',
    'simclr_pca': '#D55E00'
}

def clustering_scores(labels_true, labels_pred):
    return {
        "ACC": clustering_accuracy(labels_true, labels_pred),
        "NMI": normalized_mutual_info_score(labels_true, labels_pred),
        "ARI": clustering_accuracy(labels_true, labels_pred),
    }
    
def calculate_neighbourhood_purity(labels, W, k=10):
    n = len(labels)
    purity_scores = np.zeros(n)
    
    for i in range(n):
        neighbours = sp.find(W[i])[1]
        if len(neighbours) == 0:
            purity_scores[i] = 1.0
        else:
            neighbour_labels = labels[neighbours]
            purity_scores[i] = 1/(len(neighbours)) * np.sum(neighbour_labels == labels[i])
    
    return np.sum(purity_scores)/n, purity_scores

def bootstrap_ci(values, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    means = []
    n = len(values)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means.append(values[idx].mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return np.mean(values), lo, hi


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

def summarise(df, cols):
    n = df.shape[0]
    mean = df[cols].mean()
    std = df[cols].std(ddof=1)
    sem = std / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, df=n-1)
    ci = tcrit * sem
    return mean, ci

def bar_comparison_graphs(bars, methods=methods, labels=labels, legend=['No extra dimensions', 'Extra dimensions'], xlabel='Graph-building methodology', filename='results', ylim=(75, 100)):
    
    means, cis = [], []
    for bar in bars:
        mean, ci = summarise(bar, methods)
        means.append(mean)
        cis.append(ci)

    x = np.arange(len(methods))
    width = 0.7

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=300)

    for i, bar in enumerate(bars):
        ax.bar(x + (i+0.5)*(width/len(bars)) - width/2, 
               means[i], width/len(bars),
                yerr=cis[i], capsize=3,
                label=legend[i],
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
    
def run_iters(X, Y, methods=['knn', 'fc', 'adaptive', 'biclique', 'pcan'], params=None, kind='symmetric', extra_dims=0, labels_true=None, iters=100, num_clusters=10):
    from spectral_clustering.models.spectral import BaseSpectralClustering, PCAN
    from spectral_clustering.graphs.constructors import knn_graph, fully_connected, adaptive_neighbour_graph_can, compute_biclique_kr, epsilon_graph
    
    methods_to_fn = {
        'knn': lambda X: knn_graph(X, k=10),
        'fc': lambda X: fully_connected(X),
        'adaptive': lambda X: adaptive_neighbour_graph_can(X, k=10, symmetrise=True),
        'biclique': lambda X: compute_biclique_kr(X, k=10, r=params['r'], symmetrise=True).Kr,
        'epsilon': lambda X: epsilon_graph(X, epsilon=0.5, symmetrise=True),
    }
    
    if not isinstance(X, tuple):
        X = (X,)
    if not isinstance(Y, tuple):
        Y = (Y,)*len(X)
    if not isinstance(labels_true, tuple):
        labels_true = (labels_true,)*len(X)
    if not isinstance(num_clusters, tuple):
        num_clusters = (num_clusters,)*len(X)
        
    timings = np.zeros(shape=(len(X), len(methods)))
        
    pcan_flag = False
    results = []
    if 'pcan' in methods:
        pcan_flag = True
    for x in range(len(X)):
        k = num_clusters[x]
        if pcan_flag:        
            pcan = PCAN(n_clusters=k, k=10, lambda_=params['lambda'], kind=kind, symmetrise=True)

        spectral = BaseSpectralClustering(n_clusters=k, kind=kind)
        temp_df = pd.DataFrame(columns=methods)
        Ws = [methods_to_fn[m](X[x]) for m in methods if m != 'pcan']
        for method_idx, method in enumerate(methods):
            start = time.time()
            for i in range(iters):
                if pcan_flag and method == 'pcan':
                    labels = pcan.fit_predict(X[x])
                else:
                    W = Ws[method_idx]
                    labels = spectral.fit_predict(W, labels_true=labels_true[x], extra_dims=extra_dims)
                temp_df.loc[i, method] = clustering_accuracy(Y[x], labels)    
            timings[x, method_idx] = time.time() - start
        results.append((temp_df,))
    return results, timings

def bar_comparison(
    bars,
    methods=None,
    labels=None,
    legend=None,
    xlabel='Graph-building methodology',
    ylabel='Clustering accuracy (%)',
    filename='results',
    ylim=(0, 100),
    chance_level=None,
    overwrite_cis=None
):

    means, cis = [], []
    for bar in bars:
        mean, ci = summarise(bar, methods)
        means.append(mean)
        cis.append(ci)
    if overwrite_cis is not None:
        cis = overwrite_cis

    x = np.arange(len(methods))
    width = 0.7

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=300)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    for i, bar in enumerate(bars):
        ax.bar(
            x + (i + 0.5) * (width / len(bars)) - width / 2,
            means[i],
            width / len(bars),
            yerr=cis[i],
            capsize=3,
            label=legend[i] if legend is not None else None,
            color=[COLORS[m] for m in methods] if len(bars) == 1 else COLORS[methods[i]],
            edgecolor='black',
            linewidth=0.8,
            error_kw={'elinewidth': 0.9, 'capthick': 0.9}
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels if labels is not None else methods, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylim(ylim)

    if chance_level is not None:
        ax.axhline(
            chance_level,
            color='0.35',
            linestyle='--',
            linewidth=1.0,
            alpha=0.9
        )

    ax.grid(axis='y', linestyle='-', linewidth=0.5, color='0.85', alpha=1.0)
    ax.set_axisbelow(True)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)

    ax.tick_params(axis='both', which='major', labelsize=10, width=0.8, length=4)

    if legend is not None:
        ax.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, 1.16),
            ncol=min(2, len(legend)),
            frameon=False,
            fontsize=10
        )

    plt.tight_layout()
    plt.savefig(f"charts/{filename}.pdf", bbox_inches="tight")
    plt.savefig(f"charts/{filename}.png", bbox_inches="tight")
    plt.show()
    
def point_comparison(
    bars,
    methods=None,
    labels=None,
    legend=None,
    xlabel='Graph-building methodology',
    ylabel='Clustering accuracy (%)',
    filename='results',
    ylim=(0, 100),
    chance_level=None
):

    means, cis = [], []
    for bar in bars:
        mean, ci = summarise(bar, methods)
        means.append(mean)
        cis.append(ci)

    x = np.arange(len(methods))
    width = 0.7

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=300)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    if len(bars) == 1:
        for j, m in enumerate(methods):
            ax.errorbar(
                x[j],
                means[0][m],
                yerr=cis[0][m],
                fmt='o',
                markersize=6,
                elinewidth=1.0,
                capsize=3,
                capthick=1.0,
                color=COLORS[m],
                markerfacecolor=COLORS[m],
                markeredgecolor='black',
                markeredgewidth=0.8,
                linestyle='none',
                zorder=3
            )
    else:
        for i in range(len(bars)):
            offsets = x + (i + 0.5) * (width / len(bars)) - width / 2
            ax.errorbar(
                offsets,
                means[i],
                yerr=cis[i],
                fmt='o',
                markersize=6,
                elinewidth=1.0,
                capsize=3,
                capthick=1.0,
                color=COLORS[methods[i]],
                markerfacecolor=COLORS[methods[i]],
                markeredgecolor='black',
                markeredgewidth=0.8,
                linestyle='none',
                label=legend[i] if legend is not None else None,
                zorder=3
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels if labels is not None else methods)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_ylim(ylim)

    if chance_level is not None:
        ax.axhline(
            chance_level,
            color='0.4',
            linestyle='--',
            linewidth=1.0,
            alpha=0.9,
            zorder=1
        )

    ax.grid(axis='y', linestyle='-', linewidth=0.5, color='0.85')
    ax.set_axisbelow(True)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)

    ax.tick_params(axis='both', which='major', width=0.8, length=4)

    if legend is not None and len(bars) > 1:
        ax.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, 1.16),
            ncol=min(2, len(legend)),
            frameon=False
        )

    plt.tight_layout()
    plt.savefig(f"charts/{filename}.pdf", bbox_inches="tight")
    plt.savefig(f"charts/{filename}.png", bbox_inches="tight")
    plt.show()
    
def boxplot_with_mean(
    df,
    methods,
    labels=None,
    xlabel='Representation',
    ylabel='Clustering accuracy (%)',
    filename='cifar_knn_boxplot_mean',
    ylim=(0, 100),
    chance_level=None,
    annotate_means=False
):
    if labels is None:
        labels = methods

    data = [df[col].dropna().values for col in methods]
    means = [np.mean(d) for d in data]
    positions = np.arange(1, len(methods) + 1)

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=300)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color='black', linewidth=1.2),
        whiskerprops=dict(color='black', linewidth=0.9),
        capprops=dict(color='black', linewidth=0.9),
        boxprops=dict(edgecolor='black', linewidth=0.9)
    )

    colours = [COLORS[m] for m in methods]

    # Fill box colours
    for patch, c in zip(bp['boxes'], colours):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)

    # Overlay mean markers
    for x, m, c in zip(positions, means, colours):
        ax.scatter(
            x, m,
            s=55,
            color=c,
            edgecolor='black',
            linewidth=0.9,
            zorder=3
        )

    if annotate_means:
        for x, m in zip(positions, means):
            ax.text(
                x, m + 1.2,
                f'{m:.1f}',
                ha='center',
                va='bottom',
                fontsize=9
            )

    if chance_level is not None:
        ax.axhline(
            chance_level,
            color='0.4',
            linestyle='--',
            linewidth=1.0,
            alpha=0.9,
            zorder=1
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)

    ax.grid(axis='y', linestyle='-', linewidth=0.5, color='0.85')
    ax.set_axisbelow(True)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.tick_params(axis='both', which='major', width=0.8, length=4)

    plt.tight_layout()
    plt.savefig(f'charts/{filename}.pdf', bbox_inches='tight')
    plt.savefig(f'charts/{filename}.png', bbox_inches='tight')
    plt.show()

def prepare_line_summary(
    data,
    x_values=None,
    x_name='x',
    ci_level=0.95
):
    rows = []

    for series_label, df in data.items():
        cols = list(df.columns) if x_values is None else list(x_values)

        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Series '{series_label}' is missing columns: {missing}"
            )

        n = df.shape[0]
        alpha = 1 - ci_level
        tcrit = stats.t.ppf(1 - alpha / 2, df=n - 1) if n > 1 else np.nan

        for col in cols:
            values = df[col].dropna().to_numpy()

            n_col = len(values)
            mean = np.mean(values)
            sd = np.std(values, ddof=1) if n_col > 1 else np.nan
            sem = sd / np.sqrt(n_col) if n_col > 1 else np.nan
            ci = tcrit * sem if n_col > 1 else np.nan

            rows.append({
                x_name: col,
                'series': series_label,
                'mean': mean,
                'sd': sd,
                'sem': sem,
                'ci': ci,
                'lower': mean - ci if n_col > 1 else np.nan,
                'upper': mean + ci if n_col > 1 else np.nan,
                'n': n_col
            })

    summary_df = pd.DataFrame(rows)

    # Try to convert x to numeric if possible, so epoch plots sort correctly
    summary_df[x_name] = pd.to_numeric(summary_df[x_name])

    return summary_df

def _pick_colour(label, i, colours=None):
    if colours is not None and label in colours:
        return colours[label]

    if label in COLORS:
        return COLORS[label]

    key = str(label).strip().lower().replace('_', ' ')
    if 'raw' in key:
        return COLORS['raw']
    if 'vae' in key:
        return COLORS['vae']
    if 'simclr + pca' in key or 'simclr pca' in key or 'simclr_pca' in key:
        return COLORS['simclr_pca']
    if 'simclr' in key:
        return COLORS['simclr']


def line_comparison(
    summary_df,
    x='x',
    y='mean',
    series='series',
    lower='lower',
    upper='upper',
    xlabel='Epoch',
    ylabel='Clustering accuracy (%)',
    filename='line_results',
    ylim=None,
    xlim=None,
    colours=None,
    ci_style='band',   # 'band', 'bars', or None
    marker='o',
    markersize=4.5,
    linewidth=1.8,
    reference_line=None,
    legend=True
):
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=300)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    series_order = list(summary_df[series].drop_duplicates())

    for i, name in enumerate(series_order):
        sub = summary_df[summary_df[series] == name].sort_values(x)
        colour = _pick_colour(name, i, colours=colours)

        xvals = sub[x].to_numpy()
        yvals = sub[y].to_numpy()

        ax.plot(
            xvals,
            yvals,
            color=colour,
            marker=marker,
            markersize=markersize,
            linewidth=linewidth,
            label=name,
            zorder=3
        )

        if ci_style == 'band':
            ax.fill_between(
                xvals,
                sub[lower].to_numpy(),
                sub[upper].to_numpy(),
                color=colour,
                alpha=0.16,
                linewidth=0,
                zorder=2
            )
        elif ci_style == 'bars':
            yerr = np.vstack([
                yvals - sub[lower].to_numpy(),
                sub[upper].to_numpy() - yvals
            ])
            ax.errorbar(
                xvals,
                yvals,
                yerr=yerr,
                fmt='none',
                ecolor=colour,
                elinewidth=0.9,
                capsize=3,
                capthick=0.9,
                zorder=2
            )

    if reference_line is not None:
        ax.axhline(
            reference_line,
            color='0.35',
            linestyle='--',
            linewidth=1.0,
            alpha=0.9,
            zorder=1
        )

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)

    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)

    ax.grid(axis='y', linestyle='-', linewidth=0.5, color='0.85')
    ax.set_axisbelow(True)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)

    ax.tick_params(axis='both', which='major', labelsize=10, width=0.8, length=4)

    if legend:
        ax.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, 1.18),
            ncol=min(2, len(series_order)),
            frameon=False,
            fontsize=10
        )

    plt.tight_layout()
    plt.savefig(f"charts/{filename}.pdf", bbox_inches="tight")
    plt.savefig(f"charts/{filename}.png", bbox_inches="tight")
    plt.show()