"""
Unit tests for crypto anomaly detection.

Run with: pytest tests/ -v
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class TestFeatureRegistry:
    """Tests for feature computation."""
    
    def test_compute_volatility(self):
        """Test volatility feature computation."""
        from src.features.registry import feature_registry
        
        df = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'high': [105.0, 106.0, 107.0],
            'low': [95.0, 96.0, 97.0],
            'close': [102.0, 103.0, 104.0],
            'volume': [1000.0, 1100.0, 1200.0]
        })
        
        result = feature_registry.compute_single(df, 'volatility')
        
        # volatility = (high - low) / open
        expected = (df['high'] - df['low']) / df['open']
        
        assert result is not None
        assert len(result) == len(df)
        np.testing.assert_array_almost_equal(result.values, expected.values)
    
    def test_compute_log_return(self):
        """Test log return feature computation."""
        from src.features.registry import feature_registry
        
        df = pd.DataFrame({
            'open': [100.0, 101.0, 102.0, 103.0],
            'high': [105.0, 106.0, 107.0, 108.0],
            'low': [95.0, 96.0, 97.0, 98.0],
            'close': [100.0, 102.0, 104.0, 106.0],
            'volume': [1000.0, 1100.0, 1200.0, 1300.0]
        })
        
        result = feature_registry.compute_single(df, 'log_return')
        
        # First value should be NaN
        assert pd.isna(result.iloc[0])
        
        # Second value should be log(102/100)
        expected_second = np.log(102.0 / 100.0)
        np.testing.assert_almost_equal(result.iloc[1], expected_second, decimal=6)
    
    def test_compute_multiple_features(self):
        """Test computing multiple features at once."""
        from src.features.registry import compute_features
        
        np.random.seed(42)
        n = 100
        
        df = pd.DataFrame({
            'open': 100 + np.random.randn(n).cumsum(),
            'high': 102 + np.random.randn(n).cumsum(),
            'low': 98 + np.random.randn(n).cumsum(),
            'close': 100 + np.random.randn(n).cumsum(),
            'volume': 1000 + np.random.rand(n) * 500
        })
        
        # Ensure high >= low
        df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1) + 1
        df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1) - 1
        
        features = ['volatility', 'log_return', 'price_range']
        result = compute_features(df, features)
        
        for f in features:
            assert f in result.columns
    
    def test_dependency_resolution(self):
        """Test that feature dependencies are resolved correctly."""
        from src.features.registry import feature_registry
        
        # volume_ratio depends on volume_ma
        deps = feature_registry._resolve_dependencies(['volume_ratio'])
        
        assert 'volume_ma' in deps
        assert deps.index('volume_ma') < deps.index('volume_ratio')


class TestKMeansDetector:
    """Tests for K-Means anomaly detector."""
    
    def test_fit_and_predict(self):
        """Test basic fit and predict functionality."""
        from src.models.kmeans import KMeansAnomalyDetector
        
        np.random.seed(42)
        
        # Create clustered data
        X = np.vstack([
            np.random.randn(100, 3) + [0, 0, 0],
            np.random.randn(100, 3) + [5, 5, 5],
            np.random.randn(100, 3) + [-5, -5, -5]
        ])
        
        detector = KMeansAnomalyDetector(n_clusters=3, contamination=0.05)
        detector.fit(X)
        
        assert detector.is_fitted
        assert detector.threshold is not None
        assert detector.threshold > 0
        
        # Predict on same data
        predictions = detector.predict(X)
        
        assert len(predictions) == len(X)
        assert set(predictions).issubset({0, 1})
        
        # Anomaly rate should be close to contamination
        anomaly_rate = predictions.mean()
        assert 0.01 < anomaly_rate < 0.15
    
    def test_score_samples(self):
        """Test anomaly score computation."""
        from src.models.kmeans import KMeansAnomalyDetector
        
        np.random.seed(42)
        X = np.random.randn(100, 3)
        
        detector = KMeansAnomalyDetector(n_clusters=3)
        detector.fit(X)
        
        scores = detector.score_samples(X)
        
        assert len(scores) == len(X)
        assert all(scores >= 0)  # Distances are non-negative
    
    def test_find_optimal_k(self):
        """Test optimal k finding."""
        from src.models.kmeans import KMeansAnomalyDetector
        
        np.random.seed(42)
        
        # Create data with 3 clear clusters
        X = np.vstack([
            np.random.randn(100, 3) + [0, 0, 0],
            np.random.randn(100, 3) + [10, 10, 10],
            np.random.randn(100, 3) + [-10, -10, -10]
        ])
        
        results = KMeansAnomalyDetector.find_optimal_k(X, range(2, 8))
        
        assert 'k' in results
        assert 'inertia' in results
        assert 'silhouette' in results
        
        # Inertia should decrease as k increases
        inertias = results['inertia']
        for i in range(1, len(inertias)):
            assert inertias[i] <= inertias[i-1]


class TestGMMDetector:
    """Tests for GMM anomaly detector."""
    
    def test_fit_and_predict(self):
        """Test basic fit and predict functionality."""
        from src.models.gmm import GMMAnommalyDetector
        
        np.random.seed(42)
        X = np.random.randn(200, 3)
        
        detector = GMMAnommalyDetector(n_components=3, contamination=0.05)
        detector.fit(X)
        
        assert detector.is_fitted
        assert detector.threshold is not None
        assert detector.bic_ is not None
        assert detector.aic_ is not None
        
        predictions = detector.predict(X)
        
        assert len(predictions) == len(X)
        assert set(predictions).issubset({0, 1})
    
    def test_predict_proba_components(self):
        """Test probability predictions for components."""
        from src.models.gmm import GMMAnommalyDetector
        
        np.random.seed(42)
        X = np.random.randn(100, 3)
        
        n_components = 3
        detector = GMMAnommalyDetector(n_components=n_components)
        detector.fit(X)
        
        proba = detector.predict_proba_components(X)
        
        assert proba.shape == (len(X), n_components)
        
        # Probabilities should sum to 1
        np.testing.assert_array_almost_equal(proba.sum(axis=1), np.ones(len(X)))


class TestStreamingInference:
    """Tests for streaming inference pipeline."""
    
    def test_candle_processing(self):
        """Test processing candles through the pipeline."""
        from src.streaming.inference import StreamingInference, Candle
        from src.models.kmeans import KMeansAnomalyDetector
        
        np.random.seed(42)
        
        # Train a simple model
        X = np.random.randn(1000, 6)
        model = KMeansAnomalyDetector(n_clusters=3, contamination=0.05)
        model.fit(X, feature_names=[
            'volatility', 'log_return', 'volume_ratio',
            'return_std', 'price_range', 'volatility_ratio'
        ])
        
        inference = StreamingInference(
            model=model,
            feature_names=[
                'volatility', 'log_return', 'volume_ratio',
                'return_std', 'price_range', 'volatility_ratio'
            ],
            buffer_size=100
        )
        
        # Process enough candles to fill the buffer
        predictions = []
        base_time = datetime.now()
        
        for i in range(50):
            candle = Candle(
                time=base_time + timedelta(minutes=i),
                symbol='BTCUSDT',
                open=100 + np.random.randn(),
                high=102 + np.random.randn(),
                low=98 + np.random.randn(),
                close=100 + np.random.randn(),
                volume=1000 + np.random.rand() * 500
            )
            
            prediction = inference.process_candle(candle)
            if prediction:
                predictions.append(prediction)
        
        # Should have some predictions after buffer fills
        assert len(predictions) > 0
        
        # Check prediction structure
        for pred in predictions:
            assert hasattr(pred, 'time')
            assert hasattr(pred, 'anomaly_score')
            assert hasattr(pred, 'is_anomaly')
            assert hasattr(pred, 'threshold')


class TestDataCollector:
    """Tests for data collector (network-dependent tests are skipped)."""
    
    def test_url_building(self):
        """Test URL construction for Binance data."""
        from src.data.collector import BinanceCollector
        
        collector = BinanceCollector(market_type='spot', interval='1m')
        
        url = collector._build_monthly_url('BTCUSDT', 2024, 1)
        
        assert 'BTCUSDT' in url
        assert '2024-01' in url
        assert '1m' in url
        assert 'spot' in url
    
    def test_month_generation(self):
        """Test month list generation."""
        from src.data.collector import BinanceCollector
        
        collector = BinanceCollector()
        
        start = datetime(2024, 1, 15)
        end = datetime(2024, 4, 10)
        
        months = collector._generate_months(start, end)
        
        assert len(months) == 4
        assert months[0] == (2024, 1)
        assert months[-1] == (2024, 4)
    
    def test_dataframe_processing(self):
        """Test DataFrame processing."""
        from src.data.collector import BinanceCollector
        
        collector = BinanceCollector()
        
        # Create mock raw data
        df = pd.DataFrame({
            'open_time': [1704067200000, 1704067260000],  # Unix ms timestamps
            'open': ['100.0', '100.5'],
            'high': ['101.0', '101.5'],
            'low': ['99.0', '99.5'],
            'close': ['100.5', '101.0'],
            'volume': ['1000', '1100'],
            'close_time': [1704067259999, 1704067319999],
            'quote_volume': ['100000', '110000'],
            'trade_count': [500, 550],
            'taker_buy_volume': ['500', '550'],
            'taker_buy_quote_volume': ['50000', '55000'],
            'ignore': ['0', '0']
        })
        
        result = collector._process_dataframe(df, 'BTCUSDT')
        
        assert 'time' in result.columns
        assert 'symbol' in result.columns
        assert result['symbol'].iloc[0] == 'BTCUSDT'
        assert result['open'].dtype == float
        assert len(result) == 2


# Fixtures for pytest

@pytest.fixture
def sample_ohlcv_df():
    """Create sample OHLCV DataFrame for testing."""
    np.random.seed(42)
    n = 1000
    
    base_price = 100
    returns = np.random.randn(n) * 0.01
    prices = base_price * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'time': pd.date_range(start='2024-01-01', periods=n, freq='1min'),
        'open': prices,
        'high': prices * (1 + np.abs(np.random.randn(n) * 0.005)),
        'low': prices * (1 - np.abs(np.random.randn(n) * 0.005)),
        'close': prices * (1 + np.random.randn(n) * 0.002),
        'volume': np.abs(np.random.randn(n) * 1000 + 5000)
    })
    
    return df


@pytest.fixture
def trained_kmeans_model(sample_ohlcv_df):
    """Create a trained K-Means model."""
    from src.features.registry import get_feature_matrix
    from src.models.kmeans import KMeansAnomalyDetector
    
    features = ['volatility', 'log_return', 'price_range', 'volume_ratio', 'volatility_ratio', 'return_std']
    X, _ = get_feature_matrix(sample_ohlcv_df, features)
    
    model = KMeansAnomalyDetector(n_clusters=5, contamination=0.05)
    model.fit(X, feature_names=features)
    
    return model, features
