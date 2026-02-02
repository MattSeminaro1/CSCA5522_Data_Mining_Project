"""
Base class for anomaly detection models.

All detectors implement the same interface for consistency.
"""

from abc import ABC, abstractmethod
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Optional
import joblib
from pathlib import Path


class BaseAnomalyDetector(ABC):
    """
    Abstract base class for anomaly detectors.
    
    All detectors must implement:
    - fit(): Train on data
    - score_samples(): Return anomaly scores (higher = more anomalous)
    - predict(): Return binary anomaly labels
    """
    
    def __init__(
        self,
        contamination: float = 0.05,
        scale_features: bool = True
    ):
        """
        Initialize detector.
        
        Args:
            contamination: Expected proportion of anomalies (used to set threshold)
            scale_features: Whether to standardize features before fitting
        """
        self.contamination = contamination
        self.scale_features = scale_features
        
        self.scaler: Optional[StandardScaler] = StandardScaler() if scale_features else None
        self.threshold: Optional[float] = None
        self.feature_names: Optional[list[str]] = None
        self._is_fitted = False
    
    @property
    def is_fitted(self) -> bool:
        """Check if model has been fitted."""
        return self._is_fitted
    
    def fit(
        self,
        X: np.ndarray,
        feature_names: Optional[list[str]] = None
    ) -> 'BaseAnomalyDetector':
        """
        Fit the detector on training data.
        
        Args:
            X: Training data (n_samples, n_features)
            feature_names: Optional names for features
            
        Returns:
            self
        """
        self.feature_names = feature_names
        
        # Scale features if configured
        if self.scaler is not None:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
        
        # Fit the underlying model
        self._fit_impl(X_scaled)
        
        # Compute threshold based on contamination
        scores = self._score_impl(X_scaled)
        self.threshold = float(np.percentile(scores, (1 - self.contamination) * 100))
        
        self._is_fitted = True
        return self
    
    @abstractmethod
    def _fit_impl(self, X: np.ndarray) -> None:
        """Implementation-specific fitting logic."""
        pass
    
    @abstractmethod
    def _score_impl(self, X: np.ndarray) -> np.ndarray:
        """
        Implementation-specific scoring logic.
        
        Must return higher scores for more anomalous samples.
        """
        pass
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores for samples.
        
        Args:
            X: Data to score (n_samples, n_features)
            
        Returns:
            Array of anomaly scores (higher = more anomalous)
        """
        self._check_fitted()
        
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return self._score_impl(X_scaled)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels.
        
        Args:
            X: Data to predict (n_samples, n_features)
            
        Returns:
            Array of binary labels (1 = anomaly, 0 = normal)
        """
        scores = self.score_samples(X)
        return (scores > self.threshold).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return anomaly probabilities.
        
        Normalized scores between 0 and 1 using sigmoid transformation.
        """
        scores = self.score_samples(X)
        
        # Sigmoid centered on threshold
        normalized = 1 / (1 + np.exp(-(scores - self.threshold)))
        return normalized
    
    def _check_fitted(self) -> None:
        """Raise error if model not fitted."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before calling this method")
    
    def save(self, path: Path) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
    
    @classmethod
    def load(cls, path: Path) -> 'BaseAnomalyDetector':
        """Load model from disk."""
        return joblib.load(path)
    
    def get_params(self) -> dict:
        """Get base model parameters."""
        return {
            'contamination': self.contamination,
            'scale_features': self.scale_features,
            'threshold': self.threshold,
            'feature_names': self.feature_names
        }
    
    @abstractmethod
    def get_model_params(self) -> dict:
        """Get implementation-specific parameters."""
        pass
