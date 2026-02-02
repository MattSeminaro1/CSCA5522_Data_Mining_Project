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
tab1, tab2, tab4, tab3 = st.tabs(["Download Data", "View Data", "Export", "Explore Data"])


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
        today = datetime.utcnow().date()

        start_date_ui = st.date_input(
            "Start Date",
            value=today - timedelta(days=365),
            max_value=today
        )

        end_date_ui = st.date_input(
            "End Date",
            value=today,
            max_value=today
        )

        if start_date_ui >= end_date_ui:
            st.error("Start date must be before end date.")

        days_back = (end_date_ui - start_date_ui).days
        
        estimated_candles = len(symbols) * max(days_back, 1) * 1440
        estimated_time = max(1, len(symbols) * max(days_back, 1) // 365)
        
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
            
            end_date = datetime.combine(end_date_ui, datetime.min.time())
            start_date = datetime.combine(start_date_ui, datetime.min.time())
            
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
    st.subheader("Explore Data")

    from src.data.database import db

    explorer, schema = st.tabs(["🔎 Query Explorer", "📐 Schema"])

    # ---------------- Query Explorer ----------------

    with explorer:

        st.markdown("### Run SQL")

        default_sql = """
SELECT *
FROM raw_ohlcv
ORDER BY time DESC
LIMIT 100;
"""

        sql = st.text_area(
            "SQL Editor",
            value=default_sql,
            height=200
        )

        if st.button("Run Query", type="primary"):

            try:
                df = db.run_query(sql)

                st.success(f"{len(df):,} rows returned")
                st.dataframe(df, use_container_width=True)

            except Exception as e:
                st.error(str(e))

    # ---------------- Schema ----------------

    with schema:

        st.markdown("## Timescale Schema")

        st.markdown("""
# 📐 Database Schema (TimescaleDB)

This database contains multiple layers:

1. **Core Tables** – primary fact tables used in pipelines and ML workflows  
2. **Auxiliary / Helper Tables** – support tables, metadata, or tagging  
3. **Views & Continuous Aggregates** – derived summaries, rollups, and analytics

---

## 🧱 Core Tables

These are the main tables you should interact with:

### raw_ohlcv (Hypertable)
Primary candlestick fact table.
- PK: `(time, symbol)`
- Minute-level OHLCV, trade count, source, ingested_at
- Feeds all downstream pipelines

### features (Hypertable)
Derived feature store for ML:
- PK: `(time, symbol)`
- Contains volatility, returns, rolling stats, volume/volatility ratios
- Tracks `feature_version`

### predictions (Hypertable)
Model inference outputs:
- anomaly_score, is_anomaly, threshold_used, cluster_id
- features_json, latency_ms, predicted_at
- Indexed by anomalies and model

### collection_jobs
Tracks historical ingestion jobs:
- symbol, start_date, end_date, interval_type
- status, rows_collected, error_message, timestamps

### model_metadata
Lightweight model registry:
- model_name, model_version, hyperparameters, feature_names
- training range, evaluation metrics, is_active

---

## 🛠 Auxiliary / Helper Tables

Other tables exist for supporting workflows (may appear in dropdowns):

- `input_tags` — tags for features/models
- `audit_log` — ingestion or model audit entries
- `feature_tags` — mapping features to tags
- Any other small metadata tables  

These are **not core tables**, so definitions may not exist in this documentation.

---

## 👁 Views

- `v_data_status` — per-symbol ingestion summary  
- `v_daily_anomaly_summary` — daily anomaly aggregation  
- `v_recent_anomalies` — latest anomalies  

---

## ⚡ Continuous Aggregates (Materialized Views)

- `ohlcv_hourly` — hourly rollups  
- `ohlcv_daily` — daily rollups  
""")

        tables = db.run_query("""
SELECT table_name
FROM information_schema.tables
WHERE table_schema='public'
AND table_type='BASE TABLE'
ORDER BY table_name;
""")

        views = db.run_query("""
SELECT table_name
FROM information_schema.views
WHERE table_schema='public'
ORDER BY table_name;
""")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Tables")
            st.dataframe(tables)

        with col2:
            st.markdown("### Views")
            st.dataframe(views)

        selected = st.selectbox("Inspect Table", tables['table_name'])

        cols = db.run_query(f"""
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name='{selected}'
""")

        st.markdown("### Columns")
        st.dataframe(cols)


with tab4:
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
