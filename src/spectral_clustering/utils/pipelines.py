from spectral_clustering.graphs import fully_connected, gauss_similarity, knn_graph, gauss_knn_similarity
from spectral_clustering.models.spectral import BaseSpectralClustering
import numpy as np

def ae_then_gauss_spectral(loader, ae_trainer, n_clusters=10, N=2000):
    Z = ae_trainer.encode_dataset(loader)[:N]
    
    d = fully_connected(Z)
    S = gauss_similarity(d, np.mean(np.median(d, axis=1)))
    
    spectral = BaseSpectralClustering(n_clusters)
    return spectral.fit_predict(S)

def ae_then_gauss_knn_spectral(loader, ae_trainer, n_clusters=10, N=2000, k=10):
    Z = ae_trainer.encode_dataset(loader)[:N]
    
    d = knn_graph(Z)
    S = gauss_knn_similarity(d)
    
    spectral = BaseSpectralClustering(n_clusters)
    return spectral.fit_predict(S)