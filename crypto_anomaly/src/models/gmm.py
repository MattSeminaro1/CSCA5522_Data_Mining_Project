"""
Gaussian Mixture Model (GMM) based anomaly detector.

Anomaly score = negative log-likelihood under the mixture model.
Points with low probability under the learned distribution are anomalies.
"""

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from typing import Optional

from .base import BaseAnomalyDetector


class GMMAnommalyDetector(BaseAnomalyDetector):
    """
    Gaussian Mixture Model based anomaly detector.
    
    Fits a mixture of Gaussians to the data. Points with low
    probability under the learned distribution are anomalies.
    
    Advantages over K-Means:
    - Soft clustering (probabilistic assignments)
    - Handles elliptical clusters
    - Provides probability-based anomaly scores
    """
    
    def __init__(
        self,
        n_components: int = 5,
        contamination: float = 0.05,
        scale_features: bool = True,
        covariance_type: str = 'full',
        n_init: int = 5,
        max_iter: int = 200,
        random_state: int = 42,
        init_params: str = 'kmeans',
        reg_covar: float = 1e-6
    ):
        """
        Initialize GMM detector.

        Args:
            n_components: Number of mixture components
            contamination: Expected proportion of anomalies
            scale_features: Whether to standardize features
            covariance_type: 'full', 'tied', 'diag', or 'spherical'
            n_init: Number of random initializations
            max_iter: Maximum EM iterations
            random_state: Random seed for reproducibility
            init_params: Weight initialization method ('kmeans', 'k-means++', 'random', 'random_from_data')
            reg_covar: Regularization added to covariance diagonal for numerical stability
        """
        super().__init__(contamination, scale_features)

        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.init_params = init_params
        self.reg_covar = reg_covar

        self.model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state,
            init_params=init_params,
            reg_covar=reg_covar
        )
        
        self.bic_: Optional[float] = None
        self.aic_: Optional[float] = None
        self.silhouette_score_: Optional[float] = None
        self.cluster_labels_: Optional[np.ndarray] = None
        self._X_train: Optional[np.ndarray] = None
    
    def _fit_impl(self, X: np.ndarray) -> None:
        """Fit GMM model."""
        self._X_train = X
        self.model.fit(X)
        
        # Store metrics
        self.bic_ = float(self.model.bic(X))
        self.aic_ = float(self.model.aic(X))
        
        # Get hard cluster assignments for silhouette
        self.cluster_labels_ = self.model.predict(X)
        
        if len(X) > self.n_components and len(np.unique(self.cluster_labels_)) > 1:
            try:
                self.silhouette_score_ = float(silhouette_score(X, self.cluster_labels_))
            except ValueError:
                self.silhouette_score_ = None
    
    def _score_impl(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores as negative log-likelihood.
        
        Higher score = lower probability = more anomalous.
        """
        log_probs = self.model.score_samples(X)
        return -log_probs
    
    def predict_proba_components(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability of each sample belonging to each component.
        
        Returns:
            Array of shape (n_samples, n_components)
        """
        self._check_fitted()
        
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return self.model.predict_proba(X_scaled)
    
    def predict_cluster(self, X: np.ndarray) -> np.ndarray:
        """Predict most likely component for new data."""
        self._check_fitted()
        
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return self.model.predict(X_scaled)
    
    def get_component_means(self) -> np.ndarray:
        """Get component means in original feature space."""
        self._check_fitted()
        
        if self.scaler is not None:
            return self.scaler.inverse_transform(self.model.means_)
        return self.model.means_
    
    def get_component_weights(self) -> np.ndarray:
        """Get mixture weights (proportion of data in each component)."""
        self._check_fitted()
        return self.model.weights_
    
    def get_model_params(self) -> dict:
        """Get GMM specific parameters."""
        converged = None
        if hasattr(self.model, 'converged_'):
            converged = bool(self.model.converged_)
        
        return {
            'n_components': self.n_components,
            'covariance_type': self.covariance_type,
            'n_init': self.n_init,
            'max_iter': self.max_iter,
            'random_state': self.random_state,
            'init_params': self.init_params,
            'reg_covar': self.reg_covar,
            'bic': self.bic_,
            'aic': self.aic_,
            'silhouette_score': self.silhouette_score_,
            'converged': converged
        }
    
    @classmethod
    def find_optimal_components(
        cls,
        X: np.ndarray,
        n_range: range = range(2, 15),
        scale: bool = True,
        **kwargs
    ) -> dict:
        """
        Find optimal number of components using BIC and AIC.
        
        Args:
            X: Training data
            n_range: Range of component counts to try
            scale: Whether to standardize features
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with n values and metrics
        """
        if scale:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X
        
        results = {
            'n': [],
            'bic': [],
            'aic': [],
            'silhouette': []
        }
        
        for n in n_range:
            model = GaussianMixture(
                n_components=n,
                covariance_type=kwargs.get('covariance_type', 'full'),
                n_init=kwargs.get('n_init', 5),
                max_iter=kwargs.get('max_iter', 200),
                random_state=kwargs.get('random_state', 42)
            )
            model.fit(X_scaled)
            
            results['n'].append(n)
            results['bic'].append(float(model.bic(X_scaled)))
            results['aic'].append(float(model.aic(X_scaled)))
            
            labels = model.predict(X_scaled)
            if len(np.unique(labels)) > 1:
                try:
                    sil = float(silhouette_score(X_scaled, labels))
                    results['silhouette'].append(sil)
                except ValueError:
                    results['silhouette'].append(np.nan)
            else:
                results['silhouette'].append(np.nan)
        
        # Best by BIC (lower is better)
        results['best_bic_n'] = results['n'][int(np.argmin(results['bic']))]
        
        # Best by AIC (lower is better)
        results['best_aic_n'] = results['n'][int(np.argmin(results['aic']))]
        
        # Best by silhouette
        silhouettes = np.array(results['silhouette'])
        valid_mask = ~np.isnan(silhouettes)
        if valid_mask.any():
            results['best_silhouette_n'] = results['n'][int(np.nanargmax(silhouettes))]
        
        return results
