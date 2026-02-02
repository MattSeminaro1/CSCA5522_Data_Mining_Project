"""
Data Management Page.

Download, view, and manage historical cryptocurrency data.
"""

import streamlit as st
import pandas as pd
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import nest_asyncio

st.set_page_config(page_title="Data Management", page_icon="chart_with_upwards_trend", layout="wide")

st.title("Data Management")
st.markdown("Download and manage historical cryptocurrency data from Binance.")


nest_asyncio.apply()

def run_async(coro):
    """Run async coroutine in sync context."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# Tabs for different functions
tab1, tab2, tab3 = st.tabs(["Download Data", "View Data", "Export"])


with tab1:
    st.subheader("Download Historical Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        default_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "LINKUSDT"]
        all_symbols = default_symbols + ["BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "MATICUSDT"]
        
        symbols = st.multiselect(
            "Select Trading Pairs",
            options=all_symbols,
            default=default_symbols
        )
        
        custom_symbol = st.text_input("Or add custom symbol (e.g., DOTUSDT)")
        if custom_symbol and custom_symbol.upper() not in symbols:
            symbols.append(custom_symbol.upper())
    
    with col2:
        days_back = st.number_input(
            "Days of History",
            min_value=1,
            max_value=730,
            value=365,
            step=1,
            help="Number of days of historical data to download"
        )
        
        estimated_candles = len(symbols) * days_back * 1440
        estimated_time = max(1, len(symbols) * days_back // 365)
        
        st.info(f"""
        **Estimated download:**
        - {len(symbols)} symbols x {days_back} days
        - ~{estimated_candles:,} candles total
        - Download time: ~{estimated_time} minutes
        """)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        save_to_db = st.checkbox("Save to Database", value=True)
    with col2:
        save_to_parquet = st.checkbox("Save to Parquet", value=True)
    with col3:
        use_monthly = st.checkbox("Use Monthly Files (faster)", value=True)
        
    if st.button("Start Download", type="primary", disabled=len(symbols) == 0):
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        log_container = st.container()
        
        async def download_symbol(symbol, start_date, end_date, use_monthly_files):
            """Download a single symbol. Runs in background thread."""
            from src.data.collector import BinanceCollector
            collector = BinanceCollector()
            return await collector.collect_symbol(
                symbol, start_date, end_date, use_monthly=use_monthly_files
            )
        
        all_data = {}
        total_symbols = len(symbols)
        
        for i, symbol in enumerate(symbols):
            status_text.text(f"Downloading {symbol}... ({i+1}/{total_symbols})")
            
            end_date = datetime.utcnow() - timedelta(days=1)
            start_date = end_date - timedelta(days=days_back)
            
            try:
                df = run_async(download_symbol(symbol, start_date, end_date, use_monthly))
                
                if not df.empty:
                    all_data[symbol] = df
                    
                    with log_container:
                        st.success(f"{symbol}: {len(df):,} rows downloaded")
                    
                    if save_to_db:
                        try:
                            from src.data.database import db
                            db.ingest_ohlcv(df, symbol)
                        except Exception as e:
                            with log_container:
                                st.warning(f"DB save failed for {symbol}: {e}")
                    
                    if save_to_parquet:
                        output_dir = Path("data/raw")
                        output_dir.mkdir(parents=True, exist_ok=True)
                        df.to_parquet(output_dir / f"{symbol.lower()}.parquet")
                else:
                    with log_container:
                        st.warning(f"{symbol}: No data returned")
                        
            except Exception as e:
                with log_container:
                    st.error(f"{symbol}: {str(e)}")
            
            progress_bar.progress((i + 1) / total_symbols)
        
        status_text.text("Download complete!")
        
        total_rows = sum(len(df) for df in all_data.values())
        st.success(f"""
        **Download Complete!**
        - Symbols: {len(all_data)}
        - Total rows: {total_rows:,}
        """)


with tab2:
    st.subheader("View Loaded Data")
    
    try:
        from src.data.database import db
        
        status = db.get_data_status()
        
        if status.empty:
            st.warning("No data in database. Download some data first!")
        else:
            st.dataframe(status, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                selected_symbol = st.selectbox(
                    "Select Symbol",
                    options=status['symbol'].tolist()
                )
                
                n_rows = st.number_input(
                    "Rows to display",
                    min_value=10,
                    max_value=1000,
                    value=100
                )
            
            with col2:
                if selected_symbol:
                    recent_data = db.get_latest_candles(selected_symbol, limit=n_rows)
                    
                    if not recent_data.empty:
                        # Reverse to show oldest first for chart
                        chart_data = recent_data.iloc[::-1].copy()
                        chart_data = chart_data.set_index('time')
                        
                        st.line_chart(chart_data['close'])
                        
                        with st.expander("View Raw Data"):
                            st.dataframe(recent_data, use_container_width=True)
                    else:
                        st.info("No data found for this symbol")
                        
    except Exception as e:
        st.error(f"Could not connect to database: {e}")
        st.info("Make sure the infrastructure is running: `cd infrastructure && docker-compose up -d`")


with tab3:
    st.subheader("Export Data")
    
    try:
        from src.data.database import db
        
        status = db.get_data_status()
        
        if status.empty:
            st.warning("No data to export.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                export_symbol = st.selectbox(
                    "Symbol to Export",
                    options=status['symbol'].tolist(),
                    key="export_symbol"
                )
                
                export_format = st.selectbox(
                    "Export Format",
                    options=["Parquet", "CSV"]
                )
            
            with col2:
                symbol_data = status[status['symbol'] == export_symbol].iloc[0]
                
                min_date = pd.to_datetime(symbol_data['earliest']).date()
                max_date = pd.to_datetime(symbol_data['latest']).date()
                
                date_range = st.date_input(
                    "Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
            
            if len(date_range) == 2 and st.button("Export", type="primary"):
                with st.spinner("Exporting..."):
                    start_time = datetime.combine(date_range[0], datetime.min.time())
                    end_time = datetime.combine(date_range[1], datetime.max.time())
                    
                    df = db.get_ohlcv(export_symbol, start_time, end_time)
                    
                    if export_format == "CSV":
                        csv = df.to_csv(index=False)
                        st.download_button(
                            "Download CSV",
                            csv,
                            f"{export_symbol.lower()}_{date_range[0]}_{date_range[1]}.csv",
                            "text/csv"
                        )
                    else:
                        output_path = Path(f"/tmp/{export_symbol.lower()}.parquet")
                        df.to_parquet(output_path)
                        
                        with open(output_path, 'rb') as f:
                            st.download_button(
                                "Download Parquet",
                                f,
                                f"{export_symbol.lower()}_{date_range[0]}_{date_range[1]}.parquet",
                                "application/octet-stream"
                            )
                    
                    st.success(f"Exported {len(df):,} rows")
                    
    except Exception as e:
        st.error(f"Export failed: {e}")
