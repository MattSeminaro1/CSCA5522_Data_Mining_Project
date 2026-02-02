"""
Binance Historical Data Collector.

Downloads OHLCV data from Binance's public data repository.
Uses monthly ZIP files for efficient bulk downloads.

Data source: https://data.binance.vision/
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
import zipfile
import io
from typing import Optional, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CollectionResult:
    """Result of a data collection operation."""
    symbol: str
    rows_collected: int
    start_date: datetime
    end_date: datetime
    duration_seconds: float
    errors: list[str]


class BinanceCollector:
    """
    Downloads historical kline data from Binance's public data repository.
    
    Binance publishes daily and monthly ZIP files containing kline data.
    This collector downloads and processes these files with async I/O
    for maximum throughput.
    """
    
    BASE_URL = "https://data.binance.vision/data"
    
    KLINE_COLUMNS = [
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trade_count',
        'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'
    ]
    
    def __init__(
        self,
        market_type: str = "spot",
        interval: str = "1m",
        max_concurrent: int = 10,
        retry_attempts: int = 3
    ):
        """
        Initialize the collector.
        
        Args:
            market_type: 'spot' or 'futures'
            interval: Candle interval (1m, 5m, 15m, 1h, etc.)
            max_concurrent: Maximum concurrent downloads
            retry_attempts: Number of retry attempts for failed downloads
        """
        self.market_type = market_type
        self.interval = interval
        self.max_concurrent = max_concurrent
        self.retry_attempts = retry_attempts
        self._semaphore: Optional[asyncio.Semaphore] = None
    
    def _get_semaphore(self) -> asyncio.Semaphore:
        """Get or create semaphore for the current event loop."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent)
        return self._semaphore
    
    def _build_monthly_url(self, symbol: str, year: int, month: int) -> str:
        """Build URL for monthly kline data."""
        month_str = f"{year}-{month:02d}"
        if self.market_type == "spot":
            return (
                f"{self.BASE_URL}/spot/monthly/klines/{symbol}/"
                f"{self.interval}/{symbol}-{self.interval}-{month_str}.zip"
            )
        return (
            f"{self.BASE_URL}/futures/um/monthly/klines/{symbol}/"
            f"{self.interval}/{symbol}-{self.interval}-{month_str}.zip"
        )
    
    def _build_daily_url(self, symbol: str, date: datetime) -> str:
        """Build URL for daily kline data."""
        date_str = date.strftime("%Y-%m-%d")
        if self.market_type == "spot":
            return (
                f"{self.BASE_URL}/spot/daily/klines/{symbol}/"
                f"{self.interval}/{symbol}-{self.interval}-{date_str}.zip"
            )
        return (
            f"{self.BASE_URL}/futures/um/daily/klines/{symbol}/"
            f"{self.interval}/{symbol}-{self.interval}-{date_str}.zip"
        )
    
    async def _download_with_retry(
        self,
        session: aiohttp.ClientSession,
        url: str
    ) -> Optional[bytes]:
        """Download a URL with retry logic."""
        semaphore = self._get_semaphore()
        async with semaphore:
            for attempt in range(self.retry_attempts):
                try:
                    timeout = aiohttp.ClientTimeout(total=60)
                    async with session.get(url, timeout=timeout) as response:
                        if response.status == 404:
                            return None
                        if response.status == 200:
                            return await response.read()
                        logger.warning(
                            "Download failed: %s, status=%d, attempt=%d",
                            url, response.status, attempt + 1
                        )
                except asyncio.TimeoutError:
                    logger.warning("Download timeout: %s, attempt=%d", url, attempt + 1)
                except aiohttp.ClientError as e:
                    logger.warning("Download error: %s, %s, attempt=%d", url, e, attempt + 1)
                
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(2 ** attempt)
            
            return None
    
    def _parse_zip_content(self, content: bytes) -> Optional[pd.DataFrame]:
        """Extract and parse CSV from ZIP content."""
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                csv_name = zf.namelist()[0]
                with zf.open(csv_name) as f:
                    df = pd.read_csv(f, header=None, names=self.KLINE_COLUMNS)
                    return df
        except (zipfile.BadZipFile, IndexError, pd.errors.EmptyDataError) as e:
            logger.warning("Parse error: %s", e)
            return None
    
    async def _download_month(
        self,
        session: aiohttp.ClientSession,
        symbol: str,
        year: int,
        month: int
    ) -> Optional[pd.DataFrame]:
        """Download a single month of data."""
        url = self._build_monthly_url(symbol, year, month)
        content = await self._download_with_retry(session, url)
        if content is None:
            return None
        return self._parse_zip_content(content)
    
    async def _download_day(
        self,
        session: aiohttp.ClientSession,
        symbol: str,
        date: datetime
    ) -> Optional[pd.DataFrame]:
        """Download a single day of data."""
        url = self._build_daily_url(symbol, date)
        content = await self._download_with_retry(session, url)
        if content is None:
            return None
        return self._parse_zip_content(content)
    
    def _generate_months(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> list[tuple[int, int]]:
        """Generate list of (year, month) tuples between dates."""
        months = []
        current = start_date.replace(day=1)
        while current <= end_date:
            months.append((current.year, current.month))
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
        return months
    
    def _generate_days(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> list[datetime]:
        """Generate list of dates between start and end."""
        days = []
        current = start_date
        while current <= end_date:
            days.append(current)
            current += timedelta(days=1)
        return days
    
    def _process_dataframe(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Clean and standardize raw data."""
        if df is None or df.empty:
            return pd.DataFrame()
        
        df = df.copy()
        
        # Step 1: Convert to numeric first
        df['open_time'] = pd.to_numeric(df['open_time'], errors='coerce')
        df = df[df['open_time'].notna()]
        
        if df.empty:
            return pd.DataFrame()
        
        # Step 2: Normalize timestamp units to milliseconds.
        # Older Binance data = milliseconds (13 digits: 1_769_817_600_000)
        # Newer Binance data = microseconds (16 digits: 1_769_817_600_000_000)
        median_ts = df['open_time'].median()
        if median_ts > 1e15:
            df['open_time'] = df['open_time'] // 1000
        elif median_ts < 1e10:
            df['open_time'] = df['open_time'] * 1000
        
        # Step 3: Filter out garbage timestamps (now all in milliseconds)
        valid_min = 1_400_000_000_000
        valid_max = 2_000_000_000_000
        df = df[
            (df['open_time'] >= valid_min) &
            (df['open_time'] <= valid_max)
        ]
        
        if df.empty:
            return pd.DataFrame()
        
        # Step 4: NOW safe to convert
        df['time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
        df['symbol'] = symbol
        # Convert numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume',
                        'quote_volume', 'taker_buy_volume', 'taker_buy_quote_volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['trade_count'] = pd.to_numeric(df['trade_count'], errors='coerce').astype('Int64')
        
        # Remove duplicates and sort
        df = df.drop_duplicates(subset=['time', 'symbol'])
        df = df.sort_values('time').reset_index(drop=True)
        
        # Select output columns
        output_cols = ['time', 'symbol', 'open', 'high', 'low', 'close',
                       'volume', 'quote_volume', 'trade_count']
        return df[output_cols]
    
    async def collect_symbol(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        use_monthly: bool = True,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> pd.DataFrame:
        """
        Collect data for a single symbol.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            start_date: Start date for collection
            end_date: End date for collection
            use_monthly: Use monthly files when possible (faster)
            progress_callback: Optional callback(symbol, current, total)
            
        Returns:
            DataFrame with OHLCV data
        """
        start_time = datetime.now()
        all_dfs = []
        errors = []
        
        logger.info(
            "Starting collection: symbol=%s, start=%s, end=%s",
            symbol, start_date.date(), end_date.date()
        )
        
        connector = aiohttp.TCPConnector(limit=self.max_concurrent)
        timeout = aiohttp.ClientTimeout(total=600)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            if use_monthly:
                months = self._generate_months(start_date, end_date)
                total_units = len(months)
                
                for i, (year, month) in enumerate(months):
                    df = await self._download_month(session, symbol, year, month)
                    if df is not None and not df.empty:
                        all_dfs.append(df)
                    else:
                        errors.append(f"Failed: {year}-{month:02d}")
                    
                    if progress_callback:
                        progress_callback(symbol, i + 1, total_units)
                    
                    await asyncio.sleep(0.1)
            else:
                days = self._generate_days(start_date, end_date)
                total_units = len(days)
                batch_size = 30
                
                for i in range(0, len(days), batch_size):
                    batch_days = days[i:i + batch_size]
                    tasks = [
                        self._download_day(session, symbol, day)
                        for day in batch_days
                    ]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    for day, result in zip(batch_days, results):
                        if isinstance(result, Exception):
                            errors.append(f"Error on {day.date()}: {result}")
                        elif result is not None and not result.empty:
                            all_dfs.append(result)
                    
                    if progress_callback:
                        progress_callback(symbol, min(i + batch_size, total_units), total_units)
                    
                    await asyncio.sleep(0.2)
        
        if not all_dfs:
            logger.error("No data collected for %s", symbol)
            return pd.DataFrame()
        
        combined = pd.concat(all_dfs, ignore_index=True)
        result = self._process_dataframe(combined, symbol)
        
        # Filter to exact date range
        result = result[
            (result['time'] >= pd.Timestamp(start_date, tz='UTC')) &
            (result['time'] <= pd.Timestamp(end_date, tz='UTC'))
        ]
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(
            "Collection complete: symbol=%s, rows=%d, duration=%.1fs, errors=%d",
            symbol, len(result), duration, len(errors)
        )
        
        return result
    
    async def collect_all(
        self,
        symbols: list[str],
        days_back: int = 730,
        end_date: Optional[datetime] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> dict[str, pd.DataFrame]:
        """
        Collect data for multiple symbols.
        
        Args:
            symbols: List of trading pairs
            days_back: Number of days of history to collect
            end_date: End date (defaults to yesterday)
            progress_callback: Callback for progress updates
            
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        if end_date is None:
            end_date = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            ) - timedelta(days=1)
        
        start_date = end_date - timedelta(days=days_back)
        
        logger.info(
            "Bulk collection: symbols=%s, start=%s, end=%s",
            symbols, start_date.date(), end_date.date()
        )
        
        results = {}
        for symbol in symbols:
            df = await self.collect_symbol(
                symbol,
                start_date,
                end_date,
                progress_callback=progress_callback
            )
            results[symbol] = df
        
        total_rows = sum(len(df) for df in results.values())
        logger.info("Bulk collection complete: total_rows=%d", total_rows)
        
        return results
    
    def save_to_parquet(
        self,
        data: dict[str, pd.DataFrame],
        output_dir: Path
    ) -> dict[str, Path]:
        """Save collected data to parquet files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        paths = {}
        for symbol, df in data.items():
            if df.empty:
                continue
            
            filename = f"{symbol.lower()}.parquet"
            filepath = output_dir / filename
            df.to_parquet(filepath, index=False)
            paths[symbol] = filepath
            logger.info("Saved parquet: %s, rows=%d", filepath, len(df))
        
        return paths


def download_historical_data_sync(
    symbols: list[str],
    days_back: int = 730,
    output_dir: Optional[Path] = None
) -> dict[str, pd.DataFrame]:
    """Synchronous wrapper for data download."""
    collector = BinanceCollector()
    
    async def _run():
        data = await collector.collect_all(symbols, days_back)
        if output_dir:
            collector.save_to_parquet(data, output_dir)
        return data
    
    return asyncio.run(_run())
