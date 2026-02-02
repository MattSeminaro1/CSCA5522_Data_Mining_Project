"""
Application configuration management.

All settings can be overridden via environment variables.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Environment
    environment: str = Field(default="development")
    debug: bool = Field(default=True)
    
    # Database (TimescaleDB)
    database_url: str = Field(
        default="postgresql://crypto:crypto123@localhost:5432/crypto_anomaly"
    )
    database_pool_size: int = Field(default=10)
    database_max_overflow: int = Field(default=20)
    
    # Redis
    redis_url: str = Field(default="redis://localhost:6379")
    redis_stream_max_len: int = Field(default=10000)
    
    # MLflow
    mlflow_tracking_uri: str = Field(default="http://localhost:5001")
    mlflow_experiment_name: str = Field(default="crypto_anomaly_detection")
    
    # MinIO (S3-compatible storage)
    minio_endpoint: str = Field(default="localhost:9000")
    minio_access_key: str = Field(default="minioadmin")
    minio_secret_key: str = Field(default="minioadmin123")
    minio_secure: bool = Field(default=False)
    
    # Data collection
    symbols: list[str] = Field(
        default=["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "LINKUSDT"]
    )
    default_interval: str = Field(default="1m")
    default_lookback_days: int = Field(default=730)
    collection_batch_size: int = Field(default=30)
    
    # Feature engineering
    rolling_window: int = Field(default=10)
    default_features: list[str] = Field(
        default=[
            "volatility",
            "log_return",
            "volume_ratio",
            "return_std",
            "price_range",
            "volatility_ratio"
        ]
    )
    
    # Model training
    default_contamination: float = Field(default=0.05)
    train_test_split: float = Field(default=0.8)
    default_n_clusters: int = Field(default=5)
    
    # Streaming
    coinbase_ws_url: str = Field(default="wss://ws-feed.exchange.coinbase.com")
    stream_buffer_size: int = Field(default=100)
    prediction_log_enabled: bool = Field(default=True)
    
    # Paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
    
    @property
    def data_dir(self) -> Path:
        path = self.project_root / "data"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def raw_data_path(self) -> Path:
        path = self.data_dir / "raw"
        path.mkdir(exist_ok=True)
        return path
    
    @property
    def features_path(self) -> Path:
        path = self.data_dir / "features"
        path.mkdir(exist_ok=True)
        return path
    
    @property
    def exports_path(self) -> Path:
        path = self.data_dir / "exports"
        path.mkdir(exist_ok=True)
        return path


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
