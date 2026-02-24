"""
Model evaluation metrics and visualization.

Provides functions for analyzing model performance and generating plots.
"""

import numpy as np
import pandas as pd
from typing import Optional, Any
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import logging

logger = logging.getLogger(__name__)


def compute_clustering_metrics(
    X: np.ndarray,
    labels: np.ndarray
) -> dict[str, float]:
    """
    Compute clustering quality metrics.
    
    Args:
        X: Feature matrix
        labels: Cluster labels
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    n_clusters = len(np.unique(labels))
    if n_clusters < 2:
        logger.warning("Need at least 2 clusters for clustering metrics")
        return {'n_clusters': n_clusters}
    
    try:
        sample_size = min(2000, len(X))
        metrics['silhouette_score'] = float(silhouette_score(X, labels, sample_size=sample_size, random_state=42))
    except ValueError as e:
        logger.warning("Silhouette score failed: %s", e)
        metrics['silhouette_score'] = None
    
    try:
        metrics['calinski_harabasz_score'] = float(calinski_harabasz_score(X, labels))
    except ValueError as e:
        logger.warning("Calinski-Harabasz score failed: %s", e)
        metrics['calinski_harabasz_score'] = None
    
    try:
        metrics['davies_bouldin_score'] = float(davies_bouldin_score(X, labels))
    except ValueError as e:
        logger.warning("Davies-Bouldin score failed: %s", e)
        metrics['davies_bouldin_score'] = None
    
    metrics['n_clusters'] = n_clusters
    
    return metrics


def compute_anomaly_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: Optional[np.ndarray] = None
) -> dict[str, float]:
    """
    Compute anomaly detection metrics.
    
    Args:
        y_true: True labels (1 = anomaly, 0 = normal)
        y_pred: Predicted labels
        scores: Anomaly scores (optional, for ranking metrics)
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        precision_score, recall_score, f1_score,
        confusion_matrix, roc_auc_score
    )
    
    metrics = {}
    
    # Handle case with no true anomalies
    n_true_anomalies = y_true.sum()
    n_pred_anomalies = y_pred.sum()
    
    metrics['n_true_anomalies'] = int(n_true_anomalies)
    metrics['n_pred_anomalies'] = int(n_pred_anomalies)
    metrics['true_anomaly_rate'] = float(y_true.mean())
    metrics['pred_anomaly_rate'] = float(y_pred.mean())
    
    if n_true_anomalies == 0:
        logger.warning("No true anomalies in data, metrics may be undefined")
        return metrics
    
    try:
        metrics['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
        metrics['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
        metrics['f1_score'] = float(f1_score(y_true, y_pred, zero_division=0))
    except Exception as e:
        logger.warning("Classification metrics failed: %s", e)
    
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_positives'] = int(tp)
        metrics['false_positives'] = int(fp)
        metrics['true_negatives'] = int(tn)
        metrics['false_negatives'] = int(fn)
        
        # False positive rate
        if (fp + tn) > 0:
            metrics['false_positive_rate'] = float(fp / (fp + tn))
    except Exception as e:
        logger.warning("Confusion matrix failed: %s", e)
    
    if scores is not None:
        try:
            metrics['roc_auc'] = float(roc_auc_score(y_true, scores))
        except Exception as e:
            logger.warning("ROC AUC failed: %s", e)
    
    return metrics


def analyze_score_distribution(
    scores: np.ndarray,
    threshold: float,
    percentiles: list[int] = [1, 5, 10, 25, 50, 75, 90, 95, 99]
) -> dict[str, Any]:
    """
    Analyze the distribution of anomaly scores.
    
    Args:
        scores: Array of anomaly scores
        threshold: Current threshold
        percentiles: Percentiles to compute
        
    Returns:
        Dictionary with distribution statistics
    """
    stats = {
        'mean': float(np.mean(scores)),
        'std': float(np.std(scores)),
        'min': float(np.min(scores)),
        'max': float(np.max(scores)),
        'median': float(np.median(scores)),
        'threshold': float(threshold),
        'anomaly_rate': float((scores > threshold).mean())
    }
    
    for p in percentiles:
        stats[f'p{p}'] = float(np.percentile(scores, p))
    
    # Compute threshold percentile
    stats['threshold_percentile'] = float((scores < threshold).mean() * 100)
    
    return stats


def evaluate_model_on_period(
    model: Any,
    df: pd.DataFrame,
    feature_names: list[str],
    known_anomalies: Optional[pd.DataFrame] = None
) -> dict[str, Any]:
    """
    Evaluate model performance on a time period.
    
    Args:
        model: Trained anomaly detector
        df: DataFrame with OHLCV data
        feature_names: Features to compute
        known_anomalies: Optional DataFrame with known anomaly times
        
    Returns:
        Evaluation results
    """
    from src.features.registry import feature_registry, get_feature_matrix
    
    # Compute features
    X, df_features = get_feature_matrix(df, feature_names)
    
    # Get predictions
    scores = model.score_samples(X)
    predictions = model.predict(X)
    
    results = {
        'n_samples': len(X),
        'n_anomalies': int(predictions.sum()),
        'anomaly_rate': float(predictions.mean()),
        'score_distribution': analyze_score_distribution(scores, model.threshold)
    }
    
    # Get cluster metrics if available
    if hasattr(model, 'predict_cluster'):
        try:
            clusters = model.predict_cluster(X)
            results['clustering_metrics'] = compute_clustering_metrics(X, clusters)
        except Exception as e:
            logger.warning("Clustering metrics failed: %s", e)
    
    # If we have known anomalies, compute detection metrics
    if known_anomalies is not None:
        y_true = _create_labels_from_known(df_features, known_anomalies)
        if y_true is not None:
            results['detection_metrics'] = compute_anomaly_metrics(y_true, predictions, scores)
    
    # Find detected anomalies
    anomaly_mask = predictions == 1
    if anomaly_mask.any():
        anomaly_times = df_features.loc[anomaly_mask, 'time'].tolist() if 'time' in df_features.columns else []
        anomaly_scores = scores[anomaly_mask].tolist()
        
        results['anomalies'] = [
            {'time': t, 'score': s} 
            for t, s in zip(anomaly_times[:100], anomaly_scores[:100])
        ]
    
    return results


def _create_labels_from_known(
    df: pd.DataFrame,
    known_anomalies: pd.DataFrame,
    tolerance_minutes: int = 5
) -> Optional[np.ndarray]:
    """
    Create binary labels from known anomaly times.
    
    Args:
        df: DataFrame with 'time' column
        known_anomalies: DataFrame with 'time' column of known anomalies
        tolerance_minutes: Time window around known anomaly
        
    Returns:
        Binary label array or None if no time column
    """
    if 'time' not in df.columns or 'time' not in known_anomalies.columns:
        return None
    
    y = np.zeros(len(df), dtype=int)
    
    for anomaly_time in known_anomalies['time']:
        anomaly_time = pd.to_datetime(anomaly_time)
        
        # Mark samples within tolerance window
        time_diff = (df['time'] - anomaly_time).abs()
        mask = time_diff <= pd.Timedelta(minutes=tolerance_minutes)
        y[mask] = 1
    
    return y


def create_evaluation_report(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str]
) -> dict[str, Any]:
    """
    Create a comprehensive evaluation report.
    
    Args:
        model: Trained anomaly detector
        X_train: Training data
        X_test: Test data
        feature_names: Feature names
        
    Returns:
        Evaluation report dictionary
    """
    report = {
        'model_type': type(model).__name__,
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'model_params': model.get_model_params(),
        'threshold': model.threshold
    }
    
    # Training set analysis
    train_scores = model.score_samples(X_train)
    train_preds = model.predict(X_train)
    
    report['train'] = {
        'n_samples': len(X_train),
        'n_anomalies': int(train_preds.sum()),
        'anomaly_rate': float(train_preds.mean()),
        'score_distribution': analyze_score_distribution(train_scores, model.threshold)
    }
    
    if hasattr(model, 'predict_cluster'):
        train_clusters = model.predict_cluster(X_train)
        report['train']['clustering_metrics'] = compute_clustering_metrics(X_train, train_clusters)
    
    # Test set analysis
    test_scores = model.score_samples(X_test)
    test_preds = model.predict(X_test)
    
    report['test'] = {
        'n_samples': len(X_test),
        'n_anomalies': int(test_preds.sum()),
        'anomaly_rate': float(test_preds.mean()),
        'score_distribution': analyze_score_distribution(test_scores, model.threshold)
    }
    
    if hasattr(model, 'predict_cluster'):
        test_clusters = model.predict_cluster(X_test)
        report['test']['clustering_metrics'] = compute_clustering_metrics(X_test, test_clusters)
    
    # Feature statistics
    report['feature_stats'] = {
        'train_means': X_train.mean(axis=0).tolist(),
        'train_stds': X_train.std(axis=0).tolist(),
        'test_means': X_test.mean(axis=0).tolist(),
        'test_stds': X_test.std(axis=0).tolist()
    }
    
    return report
