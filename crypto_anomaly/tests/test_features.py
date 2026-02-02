"""
Tests for feature registry and computation.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def create_sample_ohlcv(n_rows: int = 100) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    
    base_price = 50000
    dates = [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(n_rows)]
    
    data = {
        'time': dates,
        'open': np.zeros(n_rows),
        'high': np.zeros(n_rows),
        'low': np.zeros(n_rows),
        'close': np.zeros(n_rows),
        'volume': np.abs(np.random.randn(n_rows) * 100 + 1000)
    }
    
    price = base_price
    for i in range(n_rows):
        change = np.random.randn() * 100
        data['open'][i] = price
        data['close'][i] = price + change
        data['high'][i] = max(data['open'][i], data['close'][i]) + abs(np.random.randn() * 50)
        data['low'][i] = min(data['open'][i], data['close'][i]) - abs(np.random.randn() * 50)
        price = data['close'][i]
    
    return pd.DataFrame(data)


class TestFeatureRegistry:
    """Tests for FeatureRegistry class."""
    
    def test_registry_initialization(self):
        """Test that registry initializes with default features."""
        from src.features.registry import FeatureRegistry
        
        registry = FeatureRegistry()
        features = registry.list_features()
        
        assert len(features) > 0
        assert 'volatility' in features
        assert 'log_return' in features
        assert 'volume_ratio' in features
    
    def test_get_feature(self):
        """Test retrieving a feature definition."""
        from src.features.registry import feature_registry
        
        feat = feature_registry.get('volatility')
        
        assert feat.name == 'volatility'
        assert feat.window_size == 0
        assert callable(feat.compute_fn)
    
    def test_get_nonexistent_feature(self):
        """Test that getting nonexistent feature raises KeyError."""
        from src.features.registry import feature_registry
        
        with pytest.raises(KeyError):
            feature_registry.get('nonexistent_feature')
    
    def test_compute_single_feature(self):
        """Test computing a single feature."""
        from src.features.registry import feature_registry
        
        df = create_sample_ohlcv(50)
        result = feature_registry.compute_single(df, 'volatility')
        
        assert len(result) == len(df)
        assert not result.isna().all()
    
    def test_compute_multiple_features(self):
        """Test computing multiple features at once."""
        from src.features.registry import feature_registry
        
        df = create_sample_ohlcv(50)
        features = ['volatility', 'log_return', 'price_range']
        
        result = feature_registry.compute(df, features)
        
        for feat in features:
            assert feat in result.columns
    
    def test_dependency_resolution(self):
        """Test that dependencies are computed in correct order."""
        from src.features.registry import feature_registry
        
        df = create_sample_ohlcv(50)
        
        # volume_ratio depends on volume_ma
        result = feature_registry.compute(df, ['volume_ratio'])
        
        assert 'volume_ma' in result.columns
        assert 'volume_ratio' in result.columns
    
    def test_rolling_features_have_nan_at_start(self):
        """Test that rolling features have NaN values at the beginning."""
        from src.features.registry import feature_registry
        
        df = create_sample_ohlcv(20)
        result = feature_registry.compute(df, ['return_std'])
        
        # First several values should be NaN due to rolling window
        assert result['return_std'].isna().sum() > 0


class TestComputeFeatures:
    """Tests for compute_features convenience function."""
    
    def test_compute_features_default(self):
        """Test compute_features with default feature set."""
        from src.features.registry import compute_features
        
        df = create_sample_ohlcv(50)
        result = compute_features(df)
        
        default_features = ['volatility', 'log_return', 'volume_ratio', 
                           'return_std', 'price_range', 'volatility_ratio']
        
        for feat in default_features:
            assert feat in result.columns
    
    def test_compute_features_custom(self):
        """Test compute_features with custom feature list."""
        from src.features.registry import compute_features
        
        df = create_sample_ohlcv(50)
        custom_features = ['volatility', 'body_size']
        
        result = compute_features(df, custom_features)
        
        assert 'volatility' in result.columns
        assert 'body_size' in result.columns


class TestGetFeatureMatrix:
    """Tests for get_feature_matrix function."""
    
    def test_get_feature_matrix_basic(self):
        """Test basic feature matrix extraction."""
        from src.features.registry import get_feature_matrix
        
        df = create_sample_ohlcv(100)
        features = ['volatility', 'log_return']
        
        X, df_result = get_feature_matrix(df, features)
        
        assert X.shape[1] == len(features)
        assert len(df_result) <= len(df)  # May have dropped NaN rows
    
    def test_get_feature_matrix_drops_nan(self):
        """Test that NaN rows are dropped by default."""
        from src.features.registry import get_feature_matrix
        
        df = create_sample_ohlcv(100)
        features = ['volatility', 'return_std']  # return_std has NaN at start
        
        X, df_result = get_feature_matrix(df, features, dropna=True)
        
        assert not np.isnan(X).any()
    
    def test_get_feature_matrix_keeps_nan(self):
        """Test that NaN rows are kept when dropna=False."""
        from src.features.registry import get_feature_matrix
        
        df = create_sample_ohlcv(100)
        features = ['volatility', 'return_std']
        
        X, df_result = get_feature_matrix(df, features, dropna=False)
        
        assert len(df_result) == len(df)


class TestFeatureValues:
    """Tests for correctness of computed feature values."""
    
    def test_volatility_calculation(self):
        """Test volatility is computed correctly."""
        from src.features.registry import feature_registry
        
        df = pd.DataFrame({
            'open': [100.0],
            'high': [110.0],
            'low': [90.0],
            'close': [105.0],
            'volume': [1000.0]
        })
        
        result = feature_registry.compute_single(df, 'volatility')
        
        expected = (110 - 90) / 100  # 0.2
        assert abs(result.iloc[0] - expected) < 0.0001
    
    def test_log_return_calculation(self):
        """Test log return is computed correctly."""
        from src.features.registry import feature_registry
        
        df = pd.DataFrame({
            'open': [100.0, 105.0],
            'high': [110.0, 115.0],
            'low': [90.0, 95.0],
            'close': [105.0, 110.0],
            'volume': [1000.0, 1100.0]
        })
        
        result = feature_registry.compute_single(df, 'log_return')
        
        expected = np.log(110 / 105)
        assert abs(result.iloc[1] - expected) < 0.0001
    
    def test_price_range_calculation(self):
        """Test price range is computed correctly."""
        from src.features.registry import feature_registry
        
        df = pd.DataFrame({
            'open': [100.0],
            'high': [110.0],
            'low': [90.0],
            'close': [100.0],
            'volume': [1000.0]
        })
        
        result = feature_registry.compute_single(df, 'price_range')
        
        expected = (110 - 90) / 100  # 0.2
        assert abs(result.iloc[0] - expected) < 0.0001


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
