"""
Feature Registry for Anomaly Detection.

Defines all features used for model training and inference.
Ensures consistent computation across batch and streaming contexts.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeatureDefinition:
    """
    Definition of a single feature.
    
    Attributes:
        name: Unique feature identifier
        description: Human-readable description
        compute_fn: Function that computes feature from DataFrame
        dependencies: List of feature names that must be computed first
        window_size: Rolling window size (0 for point-in-time features)
        category: Feature category for organization
    """
    name: str
    description: str
    compute_fn: Callable[[pd.DataFrame], pd.Series]
    dependencies: list[str] = field(default_factory=list)
    window_size: int = 0
    category: str = "general"


class FeatureRegistry:
    """
    Central registry of all available features.
    
    Provides consistent feature computation, dependency resolution,
    and feature metadata management.
    """
    
    def __init__(self):
        self._features: dict[str, FeatureDefinition] = {}
        self._register_default_features()
    
    def register(self, feature: FeatureDefinition) -> None:
        """Register a new feature."""
        self._features[feature.name] = feature
        logger.debug("Registered feature: %s", feature.name)
    
    def get(self, name: str) -> FeatureDefinition:
        """Get feature definition by name."""
        if name not in self._features:
            raise KeyError(f"Feature not found: {name}")
        return self._features[name]
    
    def list_features(self) -> list[str]:
        """List all registered feature names."""
        return list(self._features.keys())
    
    def list_by_category(self, category: str) -> list[str]:
        """List features in a specific category."""
        return [
            name for name, feat in self._features.items()
            if feat.category == category
        ]
    
    def get_metadata(self) -> pd.DataFrame:
        """Get metadata for all features as DataFrame."""
        data = []
        for name, feat in self._features.items():
            data.append({
                'name': name,
                'description': feat.description,
                'window_size': feat.window_size,
                'category': feat.category,
                'dependencies': ', '.join(feat.dependencies) if feat.dependencies else ''
            })
        return pd.DataFrame(data)
    
    def compute(
        self,
        df: pd.DataFrame,
        feature_names: list[str],
        validate: bool = True
    ) -> pd.DataFrame:
        """
        Compute requested features on DataFrame.
        
        Handles dependency resolution automatically.
        
        Args:
            df: Input DataFrame with OHLCV columns
            feature_names: List of features to compute
            validate: Whether to validate required columns
            
        Returns:
            DataFrame with original columns plus computed features
        """
        df = df.copy()
        
        if validate:
            required = ['open', 'high', 'low', 'close', 'volume']
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
        
        # Resolve dependencies and get computation order
        to_compute = self._resolve_dependencies(feature_names)
        computed = set(df.columns)
        
        for name in to_compute:
            if name not in computed:
                feature = self._features[name]
                try:
                    df[name] = feature.compute_fn(df)
                    computed.add(name)
                except Exception as e:
                    logger.error("Feature computation failed: %s, %s", name, e)
                    raise
        
        return df
    
    def compute_single(
        self,
        df: pd.DataFrame,
        feature_name: str
    ) -> pd.Series:
        """Compute a single feature (with dependencies)."""
        result = self.compute(df, [feature_name])
        return result[feature_name]
    
    def _resolve_dependencies(self, feature_names: list[str]) -> list[str]:
        """
        Resolve feature dependencies in correct computation order.
        
        Uses topological sort to ensure dependencies are computed first.
        """
        resolved = []
        seen = set()
        
        def visit(name: str):
            if name in seen:
                return
            seen.add(name)
            
            feature = self._features.get(name)
            if feature:
                for dep in feature.dependencies:
                    visit(dep)
                resolved.append(name)
        
        for name in feature_names:
            visit(name)
        
        return resolved
    
    def get_max_window_size(self, feature_names: list[str]) -> int:
        """Get the maximum window size across features."""
        all_features = self._resolve_dependencies(feature_names)
        window_sizes = [
            self._features[name].window_size
            for name in all_features
            if name in self._features
        ]
        return max(window_sizes) if window_sizes else 0
    
    def _register_default_features(self):
        """Register the default feature set."""
        
        # Point-in-time features (no lookback needed)
        
        self.register(FeatureDefinition(
            name="volatility",
            description="Intra-candle price swing normalized by open price",
            compute_fn=lambda df: (df['high'] - df['low']) / df['open'].replace(0, np.nan),
            dependencies=[],
            window_size=0,
            category="price"
        ))
        
        self.register(FeatureDefinition(
            name="log_return",
            description="Log return of close price",
            compute_fn=lambda df: np.log(df['close'] / df['close'].shift(1)),
            dependencies=[],
            window_size=1,
            category="price"
        ))
        
        self.register(FeatureDefinition(
            name="price_range",
            description="Normalized price spread (high-low)/close",
            compute_fn=lambda df: (df['high'] - df['low']) / df['close'].replace(0, np.nan),
            dependencies=[],
            window_size=0,
            category="price"
        ))
        
        self.register(FeatureDefinition(
            name="body_size",
            description="Candle body size relative to range",
            compute_fn=lambda df: np.abs(df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan),
            dependencies=[],
            window_size=0,
            category="price"
        ))
        
        self.register(FeatureDefinition(
            name="upper_shadow",
            description="Upper shadow relative to range",
            compute_fn=lambda df: (
                df['high'] - df[['open', 'close']].max(axis=1)
            ) / (df['high'] - df['low']).replace(0, np.nan),
            dependencies=[],
            window_size=0,
            category="price"
        ))
        
        self.register(FeatureDefinition(
            name="lower_shadow",
            description="Lower shadow relative to range",
            compute_fn=lambda df: (
                df[['open', 'close']].min(axis=1) - df['low']
            ) / (df['high'] - df['low']).replace(0, np.nan),
            dependencies=[],
            window_size=0,
            category="price"
        ))
        
        # Rolling window features
        # Note: .shift(1) ensures we only use past data to avoid look-ahead bias
        
        self.register(FeatureDefinition(
            name="volume_ma",
            description="10-period moving average of volume",
            compute_fn=lambda df: df['volume'].rolling(10, min_periods=1).mean().shift(1),
            dependencies=[],
            window_size=10,
            category="volume"
        ))
        
        self.register(FeatureDefinition(
            name="volume_ratio",
            description="Current volume vs 10-period MA",
            compute_fn=lambda df: df['volume'] / df['volume_ma'].replace(0, np.nan),
            dependencies=["volume_ma"],
            window_size=10,
            category="volume"
        ))
        
        self.register(FeatureDefinition(
            name="volatility_ma",
            description="10-period moving average of volatility",
            compute_fn=lambda df: df['volatility'].rolling(10, min_periods=1).mean().shift(1),
            dependencies=["volatility"],
            window_size=10,
            category="price"
        ))
        
        self.register(FeatureDefinition(
            name="volatility_ratio",
            description="Current volatility vs 10-period MA",
            compute_fn=lambda df: df['volatility'] / df['volatility_ma'].replace(0, np.nan),
            dependencies=["volatility", "volatility_ma"],
            window_size=10,
            category="price"
        ))
        
        self.register(FeatureDefinition(
            name="return_std",
            description="10-period rolling standard deviation of returns",
            compute_fn=lambda df: df['log_return'].rolling(10, min_periods=2).std().shift(1),
            dependencies=["log_return"],
            window_size=10,
            category="price"
        ))
        
        self.register(FeatureDefinition(
            name="return_ma",
            description="10-period moving average of returns",
            compute_fn=lambda df: df['log_return'].rolling(10, min_periods=1).mean().shift(1),
            dependencies=["log_return"],
            window_size=10,
            category="price"
        ))
        
        self.register(FeatureDefinition(
            name="return_zscore",
            description="Z-score of current return",
            compute_fn=lambda df: (
                df['log_return'] - df['return_ma']
            ) / df['return_std'].replace(0, np.nan),
            dependencies=["log_return", "return_ma", "return_std"],
            window_size=10,
            category="price"
        ))
        
        self.register(FeatureDefinition(
            name="volume_std",
            description="10-period rolling standard deviation of volume",
            compute_fn=lambda df: df['volume'].rolling(10, min_periods=2).std().shift(1),
            dependencies=[],
            window_size=10,
            category="volume"
        ))
        
        self.register(FeatureDefinition(
            name="volume_zscore",
            description="Z-score of current volume",
            compute_fn=lambda df: (
                df['volume'] - df['volume_ma']
            ) / df['volume_std'].replace(0, np.nan),
            dependencies=["volume_ma", "volume_std"],
            window_size=10,
            category="volume"
        ))
        
        self.register(FeatureDefinition(
            name="price_ma_20",
            description="20-period moving average of close price",
            compute_fn=lambda df: df['close'].rolling(20, min_periods=1).mean().shift(1),
            dependencies=[],
            window_size=20,
            category="trend"
        ))
        
        self.register(FeatureDefinition(
            name="price_position",
            description="Position of close relative to 20-period MA",
            compute_fn=lambda df: (
                df['close'] - df['price_ma_20']
            ) / df['price_ma_20'].replace(0, np.nan),
            dependencies=["price_ma_20"],
            window_size=20,
            category="trend"
        ))


# Global registry instance
feature_registry = FeatureRegistry()


def compute_features(
    df: pd.DataFrame,
    feature_names: Optional[list[str]] = None
) -> pd.DataFrame:
    """
    Compute features on a DataFrame.
    
    Args:
        df: Input DataFrame with OHLCV columns
        feature_names: Features to compute (defaults to standard set)
        
    Returns:
        DataFrame with computed features
    """
    if feature_names is None:
        feature_names = [
            "volatility",
            "log_return",
            "volume_ratio",
            "return_std",
            "price_range",
            "volatility_ratio"
        ]
    
    return feature_registry.compute(df, feature_names)


def get_feature_matrix(
    df: pd.DataFrame,
    feature_names: list[str],
    dropna: bool = True
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Get feature matrix ready for model training.
    
    Args:
        df: Input DataFrame with OHLCV columns
        feature_names: Features to include
        dropna: Whether to drop rows with NaN
        
    Returns:
        Tuple of (feature matrix, processed DataFrame)
    """
    df_features = feature_registry.compute(df, feature_names)
    
    if dropna:
        df_features = df_features.dropna(subset=feature_names)
    
    X = df_features[feature_names].values
    
    return X, df_features
