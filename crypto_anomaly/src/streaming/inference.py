"""
Streaming inference pipeline.

Handles real-time anomaly detection on incoming candle data.
Supports both live WebSocket feeds and historical replay for demos.
"""

import asyncio
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from collections import deque
from typing import Optional, Callable, Any
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class Candle:
    """Single OHLCV candle."""
    time: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    def to_dict(self) -> dict:
        return {
            'time': self.time.isoformat(),
            'symbol': self.symbol,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume
        }


@dataclass
class Prediction:
    """Anomaly prediction result."""
    time: datetime
    symbol: str
    anomaly_score: float
    is_anomaly: bool
    threshold: float
    close_price: float
    cluster_id: Optional[int] = None
    latency_ms: Optional[int] = None
    
    def to_dict(self) -> dict:
        return asdict(self)


class StreamingInference:
    """
    Real-time anomaly detection on streaming candle data.
    
    Maintains a rolling buffer of candles for feature computation
    and runs predictions on each new candle.
    """
    
    def __init__(
        self,
        model: Any,
        feature_names: list[str],
        buffer_size: int = 100
    ):
        """
        Initialize streaming inference.
        
        Args:
            model: Trained anomaly detector
            feature_names: Features to compute
            buffer_size: Number of candles to keep in buffer
        """
        self.model = model
        self.feature_names = feature_names
        self.buffer_size = buffer_size
        
        # Per-symbol buffers
        self._buffers: dict[str, deque] = {}
        
        # Import feature registry
        from src.features.registry import feature_registry
        self._feature_registry = feature_registry
        
        # Get minimum required buffer size based on feature windows
        self._min_buffer = self._feature_registry.get_max_window_size(feature_names) + 5
        
        logger.info(
            "StreamingInference initialized: features=%s, min_buffer=%d",
            feature_names, self._min_buffer
        )
    
    def _get_buffer(self, symbol: str) -> deque:
        """Get or create buffer for a symbol."""
        if symbol not in self._buffers:
            self._buffers[symbol] = deque(maxlen=self.buffer_size)
        return self._buffers[symbol]
    
    def process_candle(self, candle: Candle) -> Optional[Prediction]:
        """
        Process a single candle and return prediction.
        
        Args:
            candle: Incoming candle data
            
        Returns:
            Prediction if enough data, None otherwise
        """
        start_time = datetime.now()
        
        buffer = self._get_buffer(candle.symbol)
        buffer.append(candle)
        
        # Check if we have enough data
        if len(buffer) < self._min_buffer:
            logger.debug(
                "Insufficient buffer: symbol=%s, have=%d, need=%d",
                candle.symbol, len(buffer), self._min_buffer
            )
            return None
        
        # Convert buffer to DataFrame
        df = pd.DataFrame([
            {
                'time': c.time,
                'open': c.open,
                'high': c.high,
                'low': c.low,
                'close': c.close,
                'volume': c.volume
            }
            for c in buffer
        ])
        
        # Compute features
        try:
            df_features = self._feature_registry.compute(df, self.feature_names)
        except Exception as e:
            logger.error("Feature computation failed: %s", e)
            return None
        
        # Get the last row (current candle)
        last_row = df_features.iloc[-1]
        
        # Check for NaN values
        feature_values = [last_row.get(f) for f in self.feature_names]
        if any(pd.isna(v) for v in feature_values):
            logger.debug("NaN in features for %s", candle.symbol)
            return None
        
        # Predict
        X = np.array([feature_values])
        score = float(self.model.score_samples(X)[0])
        is_anomaly = bool(score > self.model.threshold)
        
        # Get cluster if available
        cluster_id = None
        if hasattr(self.model, 'predict_cluster'):
            try:
                cluster_id = int(self.model.predict_cluster(X)[0])
            except Exception:
                pass
        
        latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        prediction = Prediction(
            time=candle.time,
            symbol=candle.symbol,
            anomaly_score=score,
            is_anomaly=is_anomaly,
            threshold=float(self.model.threshold),
            close_price=candle.close,
            cluster_id=cluster_id,
            latency_ms=latency_ms
        )
        
        if is_anomaly:
            logger.warning(
                "ANOMALY DETECTED: symbol=%s, score=%.4f, price=%.2f",
                candle.symbol, score, candle.close
            )
        
        return prediction
    
    def clear_buffer(self, symbol: Optional[str] = None) -> None:
        """Clear buffer for a symbol or all symbols."""
        if symbol:
            if symbol in self._buffers:
                self._buffers[symbol].clear()
        else:
            self._buffers.clear()


class HistoricalReplay:
    """
    Replay historical data as a simulated stream.
    
    Useful for demonstrations and backtesting.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        speed_multiplier: float = 1.0
    ):
        """
        Initialize replay.
        
        Args:
            df: DataFrame with columns: time, symbol, open, high, low, close, volume
            speed_multiplier: Playback speed (1.0 = real-time, 10.0 = 10x faster)
        """
        required_cols = ['time', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        self.df = df.sort_values('time').reset_index(drop=True)
        self.speed_multiplier = speed_multiplier
        self._index = 0
        self._running = False
    
    async def stream(
        self,
        callback: Callable[[Candle], None],
        limit: Optional[int] = None
    ) -> None:
        """
        Stream candles with simulated timing.
        
        Args:
            callback: Function to call with each candle
            limit: Maximum number of candles to stream
        """
        self._running = True
        self._index = 0
        
        prev_time = None
        count = 0
        
        while self._running and self._index < len(self.df):
            if limit and count >= limit:
                break
            
            row = self.df.iloc[self._index]
            
            # Calculate delay based on time difference
            current_time = pd.to_datetime(row['time'])
            if prev_time is not None:
                delta = (current_time - prev_time).total_seconds()
                if delta > 0:
                    await asyncio.sleep(delta / self.speed_multiplier)
            
            candle = Candle(
                time=current_time.to_pydatetime(),
                symbol=row['symbol'],
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=float(row['volume'])
            )
            
            callback(candle)
            
            prev_time = current_time
            self._index += 1
            count += 1
        
        self._running = False
    
    def stop(self) -> None:
        """Stop the replay."""
        self._running = False
    
    def reset(self) -> None:
        """Reset to beginning."""
        self._index = 0


class CoinbaseWebSocket:
    """
    Coinbase WebSocket client for real-time candle data.
    
    Aggregates trades into candles at the specified interval.
    """
    
    WS_URL = "wss://ws-feed.exchange.coinbase.com"
    
    # Coinbase to Binance symbol mapping
    SYMBOL_MAP = {
        'BTC-USD': 'BTCUSDT',
        'ETH-USD': 'ETHUSDT',
        'SOL-USD': 'SOLUSDT',
        'AVAX-USD': 'AVAXUSDT',
        'LINK-USD': 'LINKUSDT'
    }
    
    def __init__(
        self,
        symbols: list[str],
        interval_seconds: int = 60
    ):
        """
        Initialize WebSocket client.
        
        Args:
            symbols: Coinbase product IDs (e.g., ['BTC-USD', 'ETH-USD'])
            interval_seconds: Candle interval in seconds
        """
        self.symbols = symbols
        self.interval_seconds = interval_seconds
        self._running = False
        self._candle_builders: dict[str, dict] = {}
    
    def _init_candle_builder(self, symbol: str, timestamp: datetime) -> dict:
        """Initialize a new candle builder."""
        return {
            'start_time': timestamp,
            'open': None,
            'high': float('-inf'),
            'low': float('inf'),
            'close': None,
            'volume': 0.0
        }
    
    def _process_trade(
        self,
        symbol: str,
        price: float,
        size: float,
        timestamp: datetime
    ) -> Optional[Candle]:
        """
        Process a trade and potentially emit a candle.
        
        Returns completed candle if interval has passed.
        """
        if symbol not in self._candle_builders:
            self._candle_builders[symbol] = self._init_candle_builder(symbol, timestamp)
        
        builder = self._candle_builders[symbol]
        
        # Check if interval has passed
        if (timestamp - builder['start_time']).total_seconds() >= self.interval_seconds:
            # Emit completed candle
            if builder['open'] is not None:
                completed = Candle(
                    time=builder['start_time'],
                    symbol=self.SYMBOL_MAP.get(symbol, symbol),
                    open=builder['open'],
                    high=builder['high'],
                    low=builder['low'],
                    close=builder['close'],
                    volume=builder['volume']
                )
                
                # Reset for next candle
                self._candle_builders[symbol] = self._init_candle_builder(symbol, timestamp)
                builder = self._candle_builders[symbol]
                
                # Return the completed candle
                return completed
        
        # Update current candle
        if builder['open'] is None:
            builder['open'] = price
        builder['high'] = max(builder['high'], price)
        builder['low'] = min(builder['low'], price)
        builder['close'] = price
        builder['volume'] += size
        
        return None
    
    async def stream(
        self,
        callback: Callable[[Candle], None]
    ) -> None:
        """
        Connect to WebSocket and stream candles.
        
        Args:
            callback: Function to call with each completed candle
        """
        try:
            import websockets
        except ImportError:
            logger.error("websockets package required for live streaming")
            return
        
        self._running = True
        
        subscribe_message = {
            "type": "subscribe",
            "product_ids": self.symbols,
            "channels": ["matches"]
        }
        
        while self._running:
            try:
                async with websockets.connect(self.WS_URL) as ws:
                    await ws.send(json.dumps(subscribe_message))
                    logger.info("Connected to Coinbase WebSocket")
                    
                    async for message in ws:
                        if not self._running:
                            break
                        
                        try:
                            data = json.loads(message)
                            
                            if data.get('type') == 'match':
                                symbol = data['product_id']
                                price = float(data['price'])
                                size = float(data['size'])
                                timestamp = datetime.fromisoformat(
                                    data['time'].replace('Z', '+00:00')
                                )
                                
                                candle = self._process_trade(
                                    symbol, price, size, timestamp
                                )
                                
                                if candle:
                                    callback(candle)
                                    
                        except (KeyError, ValueError) as e:
                            logger.debug("Message parse error: %s", e)
                            
            except Exception as e:
                logger.error("WebSocket error: %s", e)
                if self._running:
                    await asyncio.sleep(5)
    
    def stop(self) -> None:
        """Stop the WebSocket connection."""
        self._running = False
