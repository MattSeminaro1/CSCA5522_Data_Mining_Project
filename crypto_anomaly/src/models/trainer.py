"""
Model training with MLflow experiment tracking.

Handles hyperparameter tuning, experiment logging, and model registry.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from typing import Optional, Any
from datetime import datetime
import logging

from .base import BaseAnomalyDetector
from .kmeans import KMeansAnomalyDetector
from .gmm import GMMAnommalyDetector

logger = logging.getLogger(__name__)

# Model type registry
MODEL_REGISTRY: dict[str, type] = {
    'kmeans': KMeansAnomalyDetector,
    'gmm': GMMAnommalyDetector
}


class ModelTrainer:
    """
    Model trainer with optional MLflow experiment tracking.
    
    Features:
    - Grid search hyperparameter tuning
    - Automatic MLflow logging (when available)
    - Model comparison and selection
    """
    
    def __init__(
        self,
        mlflow_tracking_uri: Optional[str] = None,
        experiment_name: str = "crypto_anomaly_detection",
        use_mlflow: bool = True
    ):
        """
        Initialize trainer.
        
        Args:
            mlflow_tracking_uri: MLflow server URI
            experiment_name: Name of MLflow experiment
            use_mlflow: Whether to use MLflow for tracking
        """
        self.experiment_name = experiment_name
        self.use_mlflow = use_mlflow
        self._mlflow = None
        self._mlflow_available = False
        
        if use_mlflow:
            try:
                import mlflow
                import mlflow.sklearn
                self._mlflow = mlflow
                
                if mlflow_tracking_uri:
                    mlflow.set_tracking_uri(mlflow_tracking_uri)
                
                mlflow.set_experiment(experiment_name)
                self._mlflow_available = True
                logger.info("MLflow tracking enabled: %s", experiment_name)
            except ImportError:
                logger.warning("MLflow not available, training without tracking")
            except Exception as e:
                logger.warning("MLflow setup failed: %s", e)
    
    def train(
        self,
        model_type: str,
        X_train: np.ndarray,
        feature_names: list[str],
        params: Optional[dict] = None,
        X_test: Optional[np.ndarray] = None,
        tags: Optional[dict] = None,
        run_name: Optional[str] = None,
        register_model: bool = False
    ) -> tuple[BaseAnomalyDetector, Optional[str]]:
        """
        Train a single model.
        
        Args:
            model_type: 'kmeans' or 'gmm'
            X_train: Training data
            feature_names: Names of features
            params: Model parameters
            X_test: Optional test data for evaluation
            tags: Additional MLflow tags
            register_model: Whether to register model in MLflow registry
            
        Returns:
            Tuple of (trained model, MLflow run ID or None)
        """
        if model_type not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODEL_REGISTRY.keys())}")
        
        model_class = MODEL_REGISTRY[model_type]
        params = params or {}
        run_id = None
        
        # Create and train model
        model = model_class(**params)
        model.fit(X_train, feature_names=feature_names)
        
        # Compute metrics
        train_scores = model.score_samples(X_train)
        train_predictions = model.predict(X_train)
        train_anomaly_rate = float(train_predictions.mean())
        
        test_anomaly_rate = None
        if X_test is not None:
            test_scores = model.score_samples(X_test)
            test_predictions = model.predict(X_test)
            test_anomaly_rate = float(test_predictions.mean())
        
        model_params = model.get_model_params()
        
        # Log to MLflow if available
        if self._mlflow_available:
            try:
                run_id = self._log_to_mlflow(
                    model=model,
                    model_type=model_type,
                    params=params,
                    feature_names=feature_names,
                    X_train=X_train,
                    train_anomaly_rate=train_anomaly_rate,
                    train_scores=train_scores,
                    X_test=X_test,
                    test_anomaly_rate=test_anomaly_rate,
                    model_params=model_params,
                    tags=tags,
                    run_name=run_name,
                    register_model=register_model
                )
            except Exception as e:
                logger.warning("MLflow logging failed: %s", e)
        
        # Always save model locally as fallback
        try:
            from config.settings import settings
            model_dir = settings.data_dir / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            local_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
            local_path = model_dir / f"{local_id}.joblib"
            model.save(local_path)
            logger.info("Model saved locally: %s", local_path)
        except Exception as e:
            logger.warning("Local model save failed: %s", e)

        logger.info(
            "Model trained: type=%s, silhouette=%.4f, threshold=%.4f",
            model_type,
            model_params.get('silhouette_score') or 0,
            model.threshold or 0
        )

        return model, run_id
    
    def _log_to_mlflow(
        self,
        model: BaseAnomalyDetector,
        model_type: str,
        params: dict,
        feature_names: list[str],
        X_train: np.ndarray,
        train_anomaly_rate: float,
        train_scores: np.ndarray,
        X_test: Optional[np.ndarray],
        test_anomaly_rate: Optional[float],
        model_params: dict,
        tags: Optional[dict],
        run_name: Optional[str],
        register_model: bool
    ) -> str:
        """Log training run to MLflow."""
        mlflow = self._mlflow
        
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            
            # Log parameters
            mlflow.log_params(params)
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("n_features", len(feature_names))
            mlflow.log_param("n_train_samples", len(X_train))
            mlflow.set_tag("features", ",".join(feature_names))
            
            if tags:
                for key, value in tags.items():
                    mlflow.set_tag(key, str(value))
            
            # Log metrics
            if model.threshold is not None:
                mlflow.log_metric("threshold", model.threshold)
            
            for key, value in model_params.items():
                if value is not None and isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)
            
            mlflow.log_metric("train_anomaly_rate", train_anomaly_rate)
            mlflow.log_metric("train_score_mean", float(train_scores.mean()))
            mlflow.log_metric("train_score_std", float(train_scores.std()))
            
            if X_test is not None and test_anomaly_rate is not None:
                mlflow.log_metric("test_anomaly_rate", test_anomaly_rate)
                mlflow.log_param("n_test_samples", len(X_test))
            
            # Log model
            try:
                model_name = (run_name or f"{model_type}_detector") if register_model else None
                mlflow.sklearn.log_model(
                    model,
                    "model",
                    registered_model_name=model_name
                )
            except Exception as e:
                logger.warning("Model artifact logging failed (non-critical): %s", e)
            
            return run_id
    
    def train_with_tuning(
        self,
        model_type: str,
        X_train: np.ndarray,
        feature_names: list[str],
        param_grid: dict,
        feature_subsets: Optional[list[list[str]]] = None,
        X_test: Optional[np.ndarray] = None,
        metric: str = "silhouette_score",
        tags: Optional[dict] = None,
        on_progress: Optional[callable] = None
    ) -> dict:
        """
        Train with hyperparameter grid search, optionally over feature subsets.

        Args:
            model_type: 'kmeans' or 'gmm'
            X_train: Training data (columns match feature_names order)
            feature_names: Names of all features in X_train
            param_grid: Parameter grid for search
            feature_subsets: Optional list of feature name lists to search over.
                            Each subset must be a subset of feature_names.
                            If None, uses all features as a single subset.
            X_test: Optional test data
            metric: Metric to optimize
            tags: Additional MLflow tags
            on_progress: Optional callback called after each combination with
                         (current, total, result_dict)

        Returns:
            Dictionary with best model, run ID, feature subset, and all results
        """
        subsets_to_try = feature_subsets if feature_subsets else [feature_names]
        n_param_combos = len(list(ParameterGrid(param_grid)))

        logger.info(
            "Hyperparameter search: model=%s, param_combos=%d, feature_subsets=%d, total=%d",
            model_type, n_param_combos, len(subsets_to_try),
            n_param_combos * len(subsets_to_try)
        )

        results = []

        # For BIC, lower is better; for others, higher is better
        best_score = np.inf if metric == 'bic' else -np.inf
        best_model = None
        best_run_id = None
        best_features = None

        total = n_param_combos * len(subsets_to_try)
        current = 0

        for subset in subsets_to_try:
            # Select columns for this feature subset
            col_indices = [feature_names.index(f) for f in subset]
            X_train_sub = X_train[:, col_indices]
            X_test_sub = X_test[:, col_indices] if X_test is not None else None

            subset_tags = {**(tags or {}), 'feature_subset': ','.join(subset)}

            for params in ParameterGrid(param_grid):
                model, run_id = self.train(
                    model_type=model_type,
                    X_train=X_train_sub,
                    feature_names=subset,
                    params=params,
                    X_test=X_test_sub,
                    tags=subset_tags
                )

                model_params = model.get_model_params()
                score = model_params.get(metric)

                current += 1
                result_entry = None

                if score is not None:
                    result_entry = {
                        'params': params.copy(),
                        'features': list(subset),
                        'n_features': len(subset),
                        'run_id': run_id,
                        metric: score,
                        'threshold': model.threshold
                    }
                    results.append(result_entry)

                    is_better = (
                        (metric == 'bic' and score < best_score) or
                        (metric != 'bic' and score > best_score)
                    )

                    if is_better:
                        best_score = score
                        best_model = model
                        best_run_id = run_id
                        best_features = list(subset)

                if on_progress:
                    on_progress(current, total, result_entry)

        logger.info(
            "Search complete: best_%s=%.4f, best_features=%s",
            metric, best_score if best_score != np.inf else 0,
            best_features
        )

        return {
            'best_model': best_model,
            'best_run_id': best_run_id,
            'best_score': best_score if best_score != np.inf else None,
            'best_features': best_features,
            'all_results': results
        }
    
    def compare_models(
        self,
        X_train: np.ndarray,
        feature_names: list[str],
        X_test: Optional[np.ndarray] = None,
        kmeans_params: Optional[dict] = None,
        gmm_params: Optional[dict] = None
    ) -> dict:
        """
        Train and compare K-Means and GMM models.
        
        Args:
            X_train: Training data
            feature_names: Feature names
            X_test: Optional test data
            kmeans_params: K-Means parameters
            gmm_params: GMM parameters
            
        Returns:
            Comparison results
        """
        results: dict[str, Any] = {}
        
        # Train K-Means
        kmeans_model, kmeans_run_id = self.train(
            model_type='kmeans',
            X_train=X_train,
            feature_names=feature_names,
            params=kmeans_params or {},
            X_test=X_test,
            tags={'comparison': 'kmeans_vs_gmm'}
        )
        results['kmeans'] = {
            'model': kmeans_model,
            'run_id': kmeans_run_id,
            **kmeans_model.get_model_params()
        }
        
        # Train GMM
        gmm_model, gmm_run_id = self.train(
            model_type='gmm',
            X_train=X_train,
            feature_names=feature_names,
            params=gmm_params or {},
            X_test=X_test,
            tags={'comparison': 'kmeans_vs_gmm'}
        )
        results['gmm'] = {
            'model': gmm_model,
            'run_id': gmm_run_id,
            **gmm_model.get_model_params()
        }
        
        # Determine winner by silhouette score
        kmeans_sil = results['kmeans'].get('silhouette_score') or 0
        gmm_sil = results['gmm'].get('silhouette_score') or 0
        
        results['winner'] = 'kmeans' if kmeans_sil >= gmm_sil else 'gmm'
        results['winner_model'] = results[results['winner']]['model']
        
        return results
    
    def load_model(
        self,
        run_id: Optional[str] = None,
        model_name: Optional[str] = None,
        version: str = "latest"
    ) -> BaseAnomalyDetector:
        """
        Load a model from MLflow.
        
        Args:
            run_id: Specific run ID to load from
            model_name: Registered model name
            version: Model version
            
        Returns:
            Loaded model
        """
        if not self._mlflow_available:
            raise RuntimeError("MLflow not available")
        
        if run_id:
            model_uri = f"runs:/{run_id}/model"
        elif model_name:
            model_uri = f"models:/{model_name}/{version}"
        else:
            raise ValueError("Must provide either run_id or model_name")
        
        return self._mlflow.sklearn.load_model(model_uri)


def train_model(
    model_type: str,
    X_train: np.ndarray,
    feature_names: list[str],
    **kwargs
) -> BaseAnomalyDetector:
    """
    Quick function to train a model without MLflow.
    
    For development and testing.
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model_class = MODEL_REGISTRY[model_type]
    model = model_class(**kwargs)
    model.fit(X_train, feature_names=feature_names)
    
    return model


def find_best_k(
    X: np.ndarray,
    model_type: str = 'kmeans',
    k_range: range = range(2, 15)
) -> dict:
    """Find optimal number of clusters/components."""
    if model_type == 'kmeans':
        return KMeansAnomalyDetector.find_optimal_k(X, k_range)
    elif model_type == 'gmm':
        return GMMAnommalyDetector.find_optimal_components(X, k_range)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
