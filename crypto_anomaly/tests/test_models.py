"""
Tests for anomaly detection models.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile


def create_sample_data(n_samples: int = 500, n_features: int = 4) -> np.ndarray:
    """Create sample data for testing."""
    np.random.seed(42)
    
    # Create clustered data with some outliers
    n_normal = int(n_samples * 0.95)
    n_outliers = n_samples - n_normal
    
    # Normal data: 3 clusters
    cluster1 = np.random.randn(n_normal // 3, n_features) + np.array([0, 0, 0, 0])
    cluster2 = np.random.randn(n_normal // 3, n_features) + np.array([5, 5, 0, 0])
    cluster3 = np.random.randn(n_normal - 2 * (n_normal // 3), n_features) + np.array([0, 5, 5, 0])
    
    normal_data = np.vstack([cluster1, cluster2, cluster3])
    
    # Outliers: far from clusters
    outliers = np.random.randn(n_outliers, n_features) * 3 + np.array([10, 10, 10, 10])
    
    return np.vstack([normal_data, outliers])


class TestKMeansAnomalyDetector:
    """Tests for KMeansAnomalyDetector."""
    
    def test_initialization(self):
        """Test model initialization."""
        from src.models.kmeans import KMeansAnomalyDetector
        
        model = KMeansAnomalyDetector(n_clusters=5, contamination=0.05)
        
        assert model.n_clusters == 5
        assert model.contamination == 0.05
        assert not model.is_fitted
    
    def test_fit(self):
        """Test model fitting."""
        from src.models.kmeans import KMeansAnomalyDetector
        
        X = create_sample_data()
        model = KMeansAnomalyDetector(n_clusters=3)
        
        model.fit(X)
        
        assert model.is_fitted
        assert model.threshold is not None
        assert model.silhouette_score_ is not None
    
    def test_fit_with_feature_names(self):
        """Test fitting with feature names."""
        from src.models.kmeans import KMeansAnomalyDetector
        
        X = create_sample_data(n_features=4)
        feature_names = ['f1', 'f2', 'f3', 'f4']
        
        model = KMeansAnomalyDetector(n_clusters=3)
        model.fit(X, feature_names=feature_names)
        
        assert model.feature_names == feature_names
    
    def test_score_samples(self):
        """Test anomaly scoring."""
        from src.models.kmeans import KMeansAnomalyDetector
        
        X = create_sample_data()
        model = KMeansAnomalyDetector(n_clusters=3)
        model.fit(X)
        
        scores = model.score_samples(X)
        
        assert len(scores) == len(X)
        assert scores.min() >= 0  # Distances are non-negative
    
    def test_predict(self):
        """Test anomaly prediction."""
        from src.models.kmeans import KMeansAnomalyDetector
        
        X = create_sample_data()
        model = KMeansAnomalyDetector(n_clusters=3, contamination=0.05)
        model.fit(X)
        
        predictions = model.predict(X)
        
        assert len(predictions) == len(X)
        assert set(predictions).issubset({0, 1})
        
        # Anomaly rate should be approximately contamination
        anomaly_rate = predictions.mean()
        assert 0.01 < anomaly_rate < 0.15
    
    def test_predict_cluster(self):
        """Test cluster assignment."""
        from src.models.kmeans import KMeansAnomalyDetector
        
        X = create_sample_data()
        model = KMeansAnomalyDetector(n_clusters=3)
        model.fit(X)
        
        clusters = model.predict_cluster(X)
        
        assert len(clusters) == len(X)
        assert len(np.unique(clusters)) <= 3
    
    def test_get_model_params(self):
        """Test getting model parameters."""
        from src.models.kmeans import KMeansAnomalyDetector
        
        X = create_sample_data()
        model = KMeansAnomalyDetector(n_clusters=3)
        model.fit(X)
        
        params = model.get_model_params()
        
        assert 'n_clusters' in params
        assert 'silhouette_score' in params
        assert 'inertia' in params
    
    def test_save_and_load(self):
        """Test model serialization."""
        from src.models.kmeans import KMeansAnomalyDetector
        
        X = create_sample_data()
        model = KMeansAnomalyDetector(n_clusters=3)
        model.fit(X)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'model.pkl'
            model.save(path)
            
            loaded_model = KMeansAnomalyDetector.load(path)
            
            assert loaded_model.n_clusters == model.n_clusters
            assert loaded_model.threshold == model.threshold
            
            # Predictions should be identical
            np.testing.assert_array_equal(
                model.predict(X),
                loaded_model.predict(X)
            )
    
    def test_find_optimal_k(self):
        """Test finding optimal number of clusters."""
        from src.models.kmeans import KMeansAnomalyDetector
        
        X = create_sample_data()
        results = KMeansAnomalyDetector.find_optimal_k(X, k_range=range(2, 8))
        
        assert 'k' in results
        assert 'inertia' in results
        assert 'silhouette' in results
        assert len(results['k']) == 6


class TestGMMAnommalyDetector:
    """Tests for GMMAnommalyDetector."""
    
    def test_initialization(self):
        """Test model initialization."""
        from src.models.gmm import GMMAnommalyDetector
        
        model = GMMAnommalyDetector(n_components=5, contamination=0.05)
        
        assert model.n_components == 5
        assert model.contamination == 0.05
        assert not model.is_fitted
    
    def test_fit(self):
        """Test model fitting."""
        from src.models.gmm import GMMAnommalyDetector
        
        X = create_sample_data()
        model = GMMAnommalyDetector(n_components=3)
        
        model.fit(X)
        
        assert model.is_fitted
        assert model.threshold is not None
        assert model.bic_ is not None
        assert model.aic_ is not None
    
    def test_score_samples(self):
        """Test anomaly scoring."""
        from src.models.gmm import GMMAnommalyDetector
        
        X = create_sample_data()
        model = GMMAnommalyDetector(n_components=3)
        model.fit(X)
        
        scores = model.score_samples(X)
        
        assert len(scores) == len(X)
    
    def test_predict(self):
        """Test anomaly prediction."""
        from src.models.gmm import GMMAnommalyDetector
        
        X = create_sample_data()
        model = GMMAnommalyDetector(n_components=3, contamination=0.05)
        model.fit(X)
        
        predictions = model.predict(X)
        
        assert len(predictions) == len(X)
        assert set(predictions).issubset({0, 1})
    
    def test_predict_proba_components(self):
        """Test component probability prediction."""
        from src.models.gmm import GMMAnommalyDetector
        
        X = create_sample_data()
        model = GMMAnommalyDetector(n_components=3)
        model.fit(X)
        
        proba = model.predict_proba_components(X)
        
        assert proba.shape == (len(X), 3)
        
        # Each row should sum to 1
        np.testing.assert_array_almost_equal(proba.sum(axis=1), 1.0)
    
    def test_covariance_types(self):
        """Test different covariance types."""
        from src.models.gmm import GMMAnommalyDetector
        
        X = create_sample_data()
        
        for cov_type in ['full', 'tied', 'diag', 'spherical']:
            model = GMMAnommalyDetector(n_components=3, covariance_type=cov_type)
            model.fit(X)
            
            assert model.is_fitted
            assert model.covariance_type == cov_type
    
    def test_find_optimal_components(self):
        """Test finding optimal number of components."""
        from src.models.gmm import GMMAnommalyDetector
        
        X = create_sample_data()
        results = GMMAnommalyDetector.find_optimal_components(X, n_range=range(2, 6))
        
        assert 'n' in results
        assert 'bic' in results
        assert 'aic' in results
        assert 'best_bic_n' in results


class TestModelTrainer:
    """Tests for ModelTrainer."""
    
    def test_train_kmeans(self):
        """Test training K-Means model."""
        from src.models.trainer import ModelTrainer
        
        X = create_sample_data()
        feature_names = ['f1', 'f2', 'f3', 'f4']
        
        trainer = ModelTrainer(use_mlflow=False)
        model, run_id = trainer.train(
            model_type='kmeans',
            X_train=X,
            feature_names=feature_names,
            params={'n_clusters': 3}
        )
        
        assert model.is_fitted
        assert run_id is None  # No MLflow
    
    def test_train_gmm(self):
        """Test training GMM model."""
        from src.models.trainer import ModelTrainer
        
        X = create_sample_data()
        feature_names = ['f1', 'f2', 'f3', 'f4']
        
        trainer = ModelTrainer(use_mlflow=False)
        model, run_id = trainer.train(
            model_type='gmm',
            X_train=X,
            feature_names=feature_names,
            params={'n_components': 3}
        )
        
        assert model.is_fitted
    
    def test_train_invalid_model_type(self):
        """Test that invalid model type raises error."""
        from src.models.trainer import ModelTrainer
        
        X = create_sample_data()
        trainer = ModelTrainer(use_mlflow=False)
        
        with pytest.raises(ValueError):
            trainer.train(
                model_type='invalid',
                X_train=X,
                feature_names=['f1', 'f2', 'f3', 'f4']
            )
    
    def test_compare_models(self):
        """Test model comparison."""
        from src.models.trainer import ModelTrainer
        
        X = create_sample_data()
        feature_names = ['f1', 'f2', 'f3', 'f4']
        
        trainer = ModelTrainer(use_mlflow=False)
        results = trainer.compare_models(
            X_train=X,
            feature_names=feature_names,
            kmeans_params={'n_clusters': 3},
            gmm_params={'n_components': 3}
        )
        
        assert 'kmeans' in results
        assert 'gmm' in results
        assert 'winner' in results
        assert results['winner'] in ['kmeans', 'gmm']


class TestTrainModelFunction:
    """Tests for train_model convenience function."""
    
    def test_train_model(self):
        """Test train_model function."""
        from src.models.trainer import train_model
        
        X = create_sample_data()
        feature_names = ['f1', 'f2', 'f3', 'f4']
        
        model = train_model(
            model_type='kmeans',
            X_train=X,
            feature_names=feature_names,
            n_clusters=3
        )
        
        assert model.is_fitted


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
