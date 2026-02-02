"""
Prefect workflows for data collection and processing.

These flows can be scheduled and monitored via the Prefect UI.
"""

from prefect import flow, task
from prefect.tasks import task_input_hash
from datetime import datetime, timedelta
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@task(
    retries=3,
    retry_delay_seconds=60,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=1)
)
def collect_symbol_data(
    symbol: str,
    days_back: int,
    end_date: Optional[datetime] = None
) -> int:
    """
    Collect historical data for a single symbol.
    
    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        days_back: Number of days of history
        end_date: End date for collection
        
    Returns:
        Number of rows collected
    """
    import asyncio
    from src.data.collector import BinanceCollector
    from src.data.database import db
    
    if end_date is None:
        end_date = datetime.utcnow() - timedelta(days=1)
    
    start_date = end_date - timedelta(days=days_back)
    
    logger.info(f"Collecting {symbol}: {start_date.date()} to {end_date.date()}")
    
    collector = BinanceCollector()
    
    async def run_collection():
        df = await collector.collect_symbol(symbol, start_date, end_date)
        if not df.empty:
            return db.ingest_ohlcv(df, symbol)
        return 0
    
    rows = asyncio.run(run_collection())
    logger.info(f"Collected {rows:,} rows for {symbol}")
    
    return rows


@task
def compute_symbol_features(symbol: str, days: int = 30) -> int:
    """
    Compute and store features for a symbol.
    
    Args:
        symbol: Trading pair
        days: Days of data to compute features for
        
    Returns:
        Number of feature rows computed
    """
    from src.data.database import db
    from src.features.registry import compute_features
    
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days)
    
    df = db.get_ohlcv(symbol, start_time, end_time)
    
    if df.empty:
        logger.warning(f"No data found for {symbol}")
        return 0
    
    feature_names = [
        "volatility", "log_return", "volume_ratio",
        "return_std", "price_range", "volatility_ratio"
    ]
    
    df_features = compute_features(df, feature_names)
    rows = db.ingest_features(df_features, symbol)
    
    logger.info(f"Computed {rows:,} feature rows for {symbol}")
    return rows


@flow(name="collect-all-data")
def collect_all_data(
    symbols: Optional[list[str]] = None,
    days_back: int = 730
) -> dict[str, int]:
    """
    Collect historical data for all symbols.
    
    Args:
        symbols: List of symbols (uses defaults if not provided)
        days_back: Number of days of history
        
    Returns:
        Dictionary mapping symbol to rows collected
    """
    if symbols is None:
        from config.settings import settings
        symbols = settings.symbols
    
    logger.info(f"Starting collection for {len(symbols)} symbols, {days_back} days")
    
    results = {}
    for symbol in symbols:
        try:
            rows = collect_symbol_data(symbol, days_back)
            results[symbol] = rows
        except Exception as e:
            logger.error(f"Failed to collect {symbol}: {e}")
            results[symbol] = 0
    
    total = sum(results.values())
    logger.info(f"Collection complete: {total:,} total rows")
    
    return results


@flow(name="update-data")
def update_data(symbols: Optional[list[str]] = None) -> dict[str, int]:
    """
    Incremental data update - fetch only new data since last collection.
    """
    from src.data.database import db
    
    if symbols is None:
        from config.settings import settings
        symbols = settings.symbols
    
    results = {}
    
    for symbol in symbols:
        try:
            data_range = db.get_data_range(symbol)
            
            if data_range:
                _, last_time, _ = data_range
                days_to_fetch = (datetime.utcnow() - last_time).days + 1
                days_to_fetch = min(days_to_fetch, 30)
            else:
                days_to_fetch = 30
            
            if days_to_fetch > 0:
                rows = collect_symbol_data(symbol, days_to_fetch)
                results[symbol] = rows
            else:
                results[symbol] = 0
                
        except Exception as e:
            logger.error(f"Failed to update {symbol}: {e}")
            results[symbol] = 0
    
    return results


@flow(name="compute-features")
def compute_all_features(
    symbols: Optional[list[str]] = None,
    days: int = 30
) -> dict[str, int]:
    """Compute features for all symbols."""
    if symbols is None:
        from config.settings import settings
        symbols = settings.symbols
    
    results = {}
    
    for symbol in symbols:
        try:
            rows = compute_symbol_features(symbol, days)
            results[symbol] = rows
        except Exception as e:
            logger.error(f"Failed to compute features for {symbol}: {e}")
            results[symbol] = 0
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "collect":
            days = int(sys.argv[2]) if len(sys.argv) > 2 else 365
            results = collect_all_data(days_back=days)
            print(f"Collected: {results}")
            
        elif command == "update":
            results = update_data()
            print(f"Updated: {results}")
            
        elif command == "features":
            days = int(sys.argv[2]) if len(sys.argv) > 2 else 30
            results = compute_all_features(days=days)
            print(f"Features computed: {results}")
    else:
        print("Usage: python -m flows.data_collection [collect|update|features] [days]")
