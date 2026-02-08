"""
K-Means based anomaly detector.

Anomaly score = Euclidean distance to nearest cluster centroid.
Points far from all centroids are considered anomalies.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from typing import Optional

from .base import BaseAnomalyDetector


class KMeansAnomalyDetector(BaseAnomalyDetector):
    """
    K-Means clustering based anomaly detector.
    
    Clusters data into k groups. Points that are far from
    their assigned centroid are considered anomalies.
    """
    
    def __init__(
        self,
        n_clusters: int = 5,
        contamination: float = 0.05,
        scale_features: bool = True,
        n_init: int = 10,
        max_iter: int = 300,
        random_state: int = 42,
        init: str = 'k-means++',
        algorithm: str = 'lloyd'
    ):
        """
        Initialize K-Means detector.

        Args:
            n_clusters: Number of clusters (k)
            contamination: Expected proportion of anomalies
            scale_features: Whether to standardize features
            n_init: Number of random initializations
            max_iter: Maximum iterations per run
            random_state: Random seed for reproducibility
            init: Initialization method ('k-means++' or 'random')
            algorithm: K-Means algorithm ('lloyd' or 'elkan')
        """
        super().__init__(contamination, scale_features)

        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.init = init
        self.algorithm = algorithm

        self.model = KMeans(
            n_clusters=n_clusters,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state,
            init=init,
            algorithm=algorithm
        )
        
        self.cluster_labels_: Optional[np.ndarray] = None
        self.inertia_: Optional[float] = None
        self.silhouette_score_: Optional[float] = None
    
    def _fit_impl(self, X: np.ndarray) -> None:
        """Fit K-Means model."""
        self.model.fit(X)
        self.cluster_labels_ = self.model.labels_
        self.inertia_ = self.model.inertia_
        
        # Compute silhouette score if enough samples and clusters
        if len(X) > self.n_clusters and len(np.unique(self.cluster_labels_)) > 1:
            try:
                self.silhouette_score_ = float(silhouette_score(X, self.cluster_labels_))
            except ValueError:
                self.silhouette_score_ = None
    
    def _score_impl(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores as distance to nearest centroid.
        
        Higher distance = more anomalous.
        """
        distances = self.model.transform(X)
        min_distances = distances.min(axis=1)
        return min_distances
    
    def predict_cluster(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster assignment for new data."""
        self._check_fitted()
        
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return self.model.predict(X_scaled)
    
    def get_cluster_centers(self) -> np.ndarray:
        """Get cluster centroids in original feature space."""
        self._check_fitted()
        
        if self.scaler is not None:
            return self.scaler.inverse_transform(self.model.cluster_centers_)
        return self.model.cluster_centers_
    
    def get_cluster_sizes(self) -> dict[int, int]:
        """Get number of samples in each cluster."""
        self._check_fitted()
        
        unique, counts = np.unique(self.cluster_labels_, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))
    
    def get_model_params(self) -> dict:
        """Get K-Means specific parameters."""
        return {
            'n_clusters': self.n_clusters,
            'n_init': self.n_init,
            'max_iter': self.max_iter,
            'random_state': self.random_state,
            'init': self.init,
            'algorithm': self.algorithm,
            'inertia': self.inertia_,
            'silhouette_score': self.silhouette_score_
        }
    
    @classmethod
    def find_optimal_k(
        cls,
        X: np.ndarray,
        k_range: range = range(2, 15),
        scale: bool = True,
        **kwargs
    ) -> dict:
        """
        Find optimal number of clusters using elbow method and silhouette.
        
        Args:
            X: Training data
            k_range: Range of k values to try
            scale: Whether to standardize features
            **kwargs: Additional arguments for KMeansAnomalyDetector
            
        Returns:
            Dictionary with k values and metrics
        """
        if scale:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X
        
        results = {
            'k': [],
            'inertia': [],
            'silhouette': []
        }
        
        for k in k_range:
            model = KMeans(
                n_clusters=k,
                n_init=kwargs.get('n_init', 10),
                max_iter=kwargs.get('max_iter', 300),
                random_state=kwargs.get('random_state', 42)
            )
            model.fit(X_scaled)
            
            results['k'].append(k)
            results['inertia'].append(float(model.inertia_))
            
            if len(np.unique(model.labels_)) > 1:
                try:
                    sil = float(silhouette_score(X_scaled, model.labels_))
                    results['silhouette'].append(sil)
                except ValueError:
                    results['silhouette'].append(np.nan)
            else:
                results['silhouette'].append(np.nan)
        
        # Find elbow point using second derivative
        inertias = np.array(results['inertia'])
        if len(inertias) > 2:
            diffs = np.diff(inertias)
            diffs2 = np.diff(diffs)
            if len(diffs2) > 0:
                elbow_idx = int(np.argmax(np.abs(diffs2))) + 2
                results['elbow_k'] = results['k'][min(elbow_idx, len(results['k']) - 1)]
        
        # Best silhouette
        silhouettes = np.array(results['silhouette'])
        valid_mask = ~np.isnan(silhouettes)
        if valid_mask.any():
            best_sil_idx = int(np.nanargmax(silhouettes))
            results['best_silhouette_k'] = results['k'][best_sil_idx]
        
        return results
