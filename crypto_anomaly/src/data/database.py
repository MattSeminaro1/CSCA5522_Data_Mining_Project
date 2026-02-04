"""
Database operations for TimescaleDB.

Handles connection management, data ingestion, and common queries.
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
from typing import Optional, Generator, Any
from datetime import datetime
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

# Lazy import to avoid circular dependency
_settings = None


def _get_settings():
    """Lazy load settings to avoid import issues."""
    global _settings
    if _settings is None:
        try:
            from config.settings import settings
            _settings = settings
        except ImportError:
            # Fallback for when running outside container
            class FallbackSettings:
                database_url = "postgresql://crypto:crypto123@localhost:5432/crypto_anomaly"
                database_pool_size = 10
                database_max_overflow = 20
            _settings = FallbackSettings()
    return _settings


class Database:
    """
    Database connection and operations manager.
    
    Uses SQLAlchemy for connection pooling and query execution.
    Thread-safe and supports context managers.
    """
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize database connection.
        
        Args:
            database_url: PostgreSQL connection string (uses settings if not provided)
        """
        settings = _get_settings()
        self.database_url = database_url or settings.database_url
        self._engine = None
    
    @property
    def engine(self):
        """Lazy initialization of SQLAlchemy engine."""
        if self._engine is None:
            settings = _get_settings()
            self._engine = create_engine(
                self.database_url,
                poolclass=QueuePool,
                pool_size=settings.database_pool_size,
                max_overflow=settings.database_max_overflow,
                pool_pre_ping=True
            )
        return self._engine
    
    @contextmanager
    def connection(self) -> Generator:
        """Context manager for database connections."""
        conn = self.engine.connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def execute(self, query: str, params: Optional[dict] = None) -> None:
        """Execute a query without returning results."""
        with self.connection() as conn:
            conn.execute(text(query), params or {})
    
    def fetch_df(self, query: str, params: Optional[dict] = None) -> pd.DataFrame:
        """Execute query and return results as DataFrame."""
        with self.connection() as conn:
            return pd.read_sql(text(query), conn, params=params)
        
    def run_query(self, sql):
        with self.engine.begin() as conn:
            return pd.read_sql(sql, conn)
    
    def fetch_one(self, query: str, params: Optional[dict] = None) -> Optional[tuple]:
        """Execute query and return single row."""
        with self.connection() as conn:
            result = conn.execute(text(query), params or {})
            return result.fetchone()
    
    def test_connection(self) -> bool:
        """Test if database connection is working."""
        try:
            with self.connection() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error("Database connection failed: %s", e)
            return False
    
    # Data Ingestion Methods
    
    def ingest_ohlcv(
        self,
        df: pd.DataFrame,
        symbol: str,
        batch_size: int = 10000
    ) -> int:
        """
        Ingest OHLCV data into raw_ohlcv table.
        
        Args:
            df: DataFrame with columns: time, open, high, low, close, volume
            symbol: Trading pair symbol
            batch_size: Rows per batch insert
            
        Returns:
            Number of rows inserted
        """
        if df.empty:
            return 0
        
        required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        df = df.copy()
        
        # Ensure symbol column exists
        if 'symbol' not in df.columns:
            df['symbol'] = symbol
        
        # Select columns that exist in the dataframe
        db_cols = ['time', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        optional_cols = ['quote_volume', 'trade_count']
        for col in optional_cols:
            if col in df.columns:
                db_cols.append(col)
        
        insert_df = df[db_cols].copy()
        
        # Handle duplicates by keeping first occurrence
        insert_df = insert_df.drop_duplicates(subset=['time', 'symbol'], keep='first')
        
        total_inserted = 0
        
        for i in range(0, len(insert_df), batch_size):
            batch = insert_df.iloc[i:i + batch_size]
            
            try:
                # Use upsert pattern with ON CONFLICT
                self._upsert_batch(batch, 'raw_ohlcv', ['time', 'symbol'])
                total_inserted += len(batch)
            except Exception as e:
                logger.error("Batch insert failed: %s", e)
                raise
        
        logger.info("Ingested OHLCV: symbol=%s, rows=%d", symbol, total_inserted)
        return total_inserted
    
    def _upsert_batch(
        self,
        df: pd.DataFrame,
        table_name: str,
        conflict_columns: list[str]
    ) -> None:
        """Perform upsert (insert or update on conflict)."""
        if df.empty:
            return
        
        columns = df.columns.tolist()
        col_str = ', '.join(columns)
        
        # Create placeholders
        placeholders = ', '.join([f':{col}' for col in columns])
        
        # Create update clause for non-conflict columns
        update_cols = [c for c in columns if c not in conflict_columns]
        update_str = ', '.join([f'{c} = EXCLUDED.{c}' for c in update_cols])
        
        conflict_str = ', '.join(conflict_columns)
        
        query = f"""
            INSERT INTO {table_name} ({col_str})
            VALUES ({placeholders})
            ON CONFLICT ({conflict_str}) DO UPDATE SET {update_str}
        """
        
        with self.connection() as conn:
            for _, row in df.iterrows():
                params = {col: self._convert_value(row[col]) for col in columns}
                conn.execute(text(query), params)
    
    def _convert_value(self, value: Any) -> Any:
        """Convert pandas/numpy types to Python native types."""
        if pd.isna(value):
            return None
        if isinstance(value, (np.integer, np.int64)):
            return int(value)
        if isinstance(value, (np.floating, np.float64)):
            return float(value)
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime()
        return value
    
    def ingest_features(
        self,
        df: pd.DataFrame,
        symbol: str,
        feature_version: str = "v1"
    ) -> int:
        """Ingest computed features into features table."""
        if df.empty:
            return 0
        
        df = df.copy()
        df['symbol'] = symbol
        df['feature_version'] = feature_version
        
        # Select only columns that exist in the features table
        valid_cols = [
            'time', 'symbol', 'volatility', 'log_return', 'price_range',
            'volume_ratio', 'volatility_ratio', 'return_std',
            'volume_ma', 'volatility_ma', 'feature_version'
        ]
        available_cols = [c for c in valid_cols if c in df.columns]
        insert_df = df[available_cols].dropna(subset=['time'])
        
        self._upsert_batch(insert_df, 'features', ['time', 'symbol'])
        
        logger.info("Ingested features: symbol=%s, rows=%d", symbol, len(insert_df))
        return len(insert_df)
    
    def log_prediction(
        self,
        time: datetime,
        symbol: str,
        model_name: str,
        model_version: str,
        anomaly_score: float,
        is_anomaly: bool,
        threshold: float,
        close_price: Optional[float] = None,
        features_json: Optional[dict] = None,
        latency_ms: Optional[int] = None,
        source: str = "streaming"
    ) -> None:
        """Log a prediction to the predictions table."""
        query = """
            INSERT INTO predictions 
            (time, symbol, model_name, model_version, anomaly_score, 
             is_anomaly, threshold_used, close_price, features_json, 
             latency_ms, source)
            VALUES 
            (:time, :symbol, :model_name, :model_version, :anomaly_score,
             :is_anomaly, :threshold, :close_price, :features_json,
             :latency_ms, :source)
        """
        
        self.execute(query, {
            'time': time,
            'symbol': symbol,
            'model_name': model_name,
            'model_version': model_version,
            'anomaly_score': float(anomaly_score),
            'is_anomaly': bool(is_anomaly),
            'threshold': float(threshold),
            'close_price': float(close_price) if close_price else None,
            'features_json': json.dumps(features_json) if features_json else None,
            'latency_ms': int(latency_ms) if latency_ms else None,
            'source': source
        })
    
    # Query Methods
    
    def get_data_status(self) -> pd.DataFrame:
        """Get data availability status per symbol."""
        query = """
        SET timescaledb.enable_vectorized_aggregation = off;
        SELECT 
            symbol,
            MIN(time) as earliest,
            MAX(time) as latest,
            COUNT(*) as total_rows,
            MAX(time) - MIN(time) as time_span
        FROM raw_ohlcv
        GROUP BY symbol
        ORDER BY symbol
        """
        return pd.read_sql(query, self.engine)
    
    def get_data_range(self, symbol: str) -> Optional[tuple]:
        """Get the time range of data available for a symbol."""
        result = self.fetch_one("""
            SELECT MIN(time), MAX(time), COUNT(*)
            FROM raw_ohlcv
            WHERE symbol = :symbol
        """, {'symbol': symbol})
        
        if result and result[0]:
            return (result[0], result[1], result[2])
        return None
    
    def get_ohlcv(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """Get OHLCV data for a symbol and time range."""
        return self.fetch_df("""
            SELECT time, open, high, low, close, volume, quote_volume, trade_count
            FROM raw_ohlcv
            WHERE symbol = :symbol
              AND time >= :start_time
              AND time <= :end_time
            ORDER BY time
        """, {
            'symbol': symbol,
            'start_time': start_time,
            'end_time': end_time
        })
    
    def get_latest_candles(
        self,
        symbol: str,
        limit: int = 100
    ) -> pd.DataFrame:
        """Get the most recent candles for a symbol."""
        return self.fetch_df("""
            SELECT time, open, high, low, close, volume
            FROM raw_ohlcv
            WHERE symbol = :symbol
            ORDER BY time DESC
            LIMIT :limit
        """, {'symbol': symbol, 'limit': limit})
    
    def get_features(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        feature_names: Optional[list[str]] = None
    ) -> pd.DataFrame:
        """Get computed features for a symbol and time range."""
        if feature_names:
            cols = ', '.join(['time'] + feature_names)
        else:
            cols = '*'
        
        query = f"""
            SELECT {cols}
            FROM features
            WHERE symbol = :symbol
              AND time >= :start_time
              AND time <= :end_time
            ORDER BY time
        """
        
        return self.fetch_df(query, {
            'symbol': symbol,
            'start_time': start_time,
            'end_time': end_time
        })
    
    def get_predictions(
        self,
        symbol: str,
        model_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        anomalies_only: bool = False,
        limit: int = 1000
    ) -> pd.DataFrame:
        """Get predictions with optional filters."""
        conditions = ["symbol = :symbol"]
        params: dict[str, Any] = {'symbol': symbol, 'limit': limit}
        
        if model_name:
            conditions.append("model_name = :model_name")
            params['model_name'] = model_name
        
        if start_time:
            conditions.append("time >= :start_time")
            params['start_time'] = start_time
        
        if end_time:
            conditions.append("time <= :end_time")
            params['end_time'] = end_time
        
        if anomalies_only:
            conditions.append("is_anomaly = TRUE")
        
        where_clause = " AND ".join(conditions)
        
        return self.fetch_df(f"""
            SELECT time, model_name, model_version, anomaly_score, 
                   is_anomaly, threshold_used, close_price, latency_ms
            FROM predictions
            WHERE {where_clause}
            ORDER BY time DESC
            LIMIT :limit
        """, params)
    
    def get_recent_anomalies(
        self,
        hours: int = 24,
        limit: int = 100
    ) -> pd.DataFrame:
        """Get recent anomalies across all symbols."""
        return self.fetch_df("""
            SELECT time, symbol, model_name, anomaly_score, 
                   threshold_used, close_price
            FROM predictions
            WHERE is_anomaly = TRUE
              AND time > NOW() - INTERVAL '1 hour' * :hours
            ORDER BY time DESC
            LIMIT :limit
        """, {'hours': hours, 'limit': limit})
    
    def get_anomaly_summary(
        self,
        symbol: str,
        days: int = 7
    ) -> pd.DataFrame:
        """Get daily anomaly summary for a symbol."""
        return self.fetch_df("""
            SELECT * FROM v_daily_anomaly_summary
            WHERE symbol = :symbol
              AND day > NOW() - INTERVAL '1 day' * :days
            ORDER BY day DESC
        """, {'symbol': symbol, 'days': days})
    
    # Export Methods
    
    def export_to_parquet(
        self,
        symbol: str,
        output_path: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> str:
        """Export OHLCV data to parquet file."""
        conditions = ["symbol = :symbol"]
        params: dict[str, Any] = {'symbol': symbol}
        
        if start_time:
            conditions.append("time >= :start_time")
            params['start_time'] = start_time
        
        if end_time:
            conditions.append("time <= :end_time")
            params['end_time'] = end_time
        
        where_clause = " AND ".join(conditions)
        
        df = self.fetch_df(f"""
            SELECT time, symbol, open, high, low, close, volume, 
                   quote_volume, trade_count
            FROM raw_ohlcv
            WHERE {where_clause}
            ORDER BY time
        """, params)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        
        logger.info("Exported parquet: %s, rows=%d", output_path, len(df))
        return str(output_path)


# Global database instance
db = Database()
