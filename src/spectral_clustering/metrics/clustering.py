from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

def clustering_scores(labels_true, labels_pred):
    return {
        "NMI": normalized_mutual_info_score(labels_true, labels_pred),
        "ARI": adjusted_rand_score(labels_true, labels_pred),
    }
