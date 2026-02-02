"""
Live Detection Page.

Real-time anomaly detection with streaming data replay.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from collections import deque
import time

st.set_page_config(page_title="Live Detection", page_icon="chart_with_upwards_trend", layout="wide")

st.title("Live Detection")
st.markdown("Real-time anomaly detection on streaming cryptocurrency data.")


def get_available_symbols() -> list[str]:
    """Get list of symbols with data in database."""
    try:
        from src.data.database import db
        status = db.get_data_status()
        if not status.empty:
            return status['symbol'].tolist()
    except Exception:
        pass
    return []


def get_mlflow_models() -> tuple[list[str], dict[str, str]]:
    """Get available models from MLflow."""
    model_options = []
    run_id_map = {}
    
    try:
        import mlflow
        from config.settings import settings
        
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        
        experiments = mlflow.search_experiments()
        for exp in experiments:
            if exp.name != "Default":
                runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
                for _, run in runs.iterrows():
                    model_type = run.get('params.model_type', 'unknown')
                    run_id = run['run_id']
                    sil = run.get('metrics.silhouette_score', 0)
                    if pd.notna(sil):
                        label = f"{model_type} - {run_id[:8]} (sil: {sil:.3f})"
                    else:
                        label = f"{model_type} - {run_id[:8]}"
                    model_options.append(label)
                    run_id_map[label] = run_id
    except Exception:
        pass
    
    return model_options, run_id_map


def load_model_from_mlflow(run_id: str):
    """Load a model from MLflow by run ID."""
    import mlflow.sklearn
    return mlflow.sklearn.load_model(f"runs:/{run_id}/model")


model_options, run_id_map = get_mlflow_models()

# Tabs
tab1, tab2 = st.tabs(["Historical Replay", "Live Stream (Demo)"])


with tab1:
    st.subheader("Historical Data Replay")
    
    st.markdown("""
    Replay historical data to simulate real-time anomaly detection.
    This demonstrates how the system would work with live data.
    """)
    
    if not model_options:
        st.warning("No trained models available. Train a model first.")
    else:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_model = st.selectbox("Select Model", options=model_options, key="replay_model")
        
        with col2:
            available_symbols = get_available_symbols()
            replay_symbol = st.selectbox(
                "Symbol",
                options=available_symbols if available_symbols else ["BTCUSDT"],
                key="replay_symbol"
            )
        
        with col3:
            replay_speed = st.slider("Replay Speed (candles/sec)", min_value=1, max_value=50, value=10, key="replay_speed")
        
        replay_minutes = st.slider("Minutes to Replay", min_value=60, max_value=1440, value=240, key="replay_minutes")
        
        if 'replay_running' not in st.session_state:
            st.session_state.replay_running = False
        if 'replay_data' not in st.session_state:
            st.session_state.replay_data = {
                'times': deque(maxlen=500),
                'prices': deque(maxlen=500),
                'scores': deque(maxlen=500),
                'anomalies': deque(maxlen=500)
            }
        
        col1, col2 = st.columns(2)
        
        with col1:
            start_button = st.button("Start Replay", type="primary", disabled=st.session_state.replay_running)
        with col2:
            stop_button = st.button("Stop Replay", disabled=not st.session_state.replay_running)
        
        if stop_button:
            st.session_state.replay_running = False
        
        # Metrics display
        metric_cols = st.columns(4)
        
        with metric_cols[0]:
            candles_placeholder = st.empty()
        with metric_cols[1]:
            anomalies_placeholder = st.empty()
        with metric_cols[2]:
            rate_placeholder = st.empty()
        with metric_cols[3]:
            latency_placeholder = st.empty()
        
        # Chart placeholder
        chart_placeholder = st.empty()
        
        # Recent anomalies table
        anomaly_table_placeholder = st.empty()
        
        if start_button:
            st.session_state.replay_running = True
            
            # Clear previous data
            for key in st.session_state.replay_data:
                st.session_state.replay_data[key].clear()
            
            try:
                from src.data.database import db
                from src.streaming.inference import StreamingInference, Candle
                
                # Load model
                run_id = run_id_map[selected_model]
                model = load_model_from_mlflow(run_id)
                
                feature_names = model.feature_names or [
                    "volatility", "log_return", "volume_ratio",
                    "return_std", "price_range", "volatility_ratio"
                ]
                
                # Initialize streaming inference
                inference = StreamingInference(
                    model=model,
                    feature_names=feature_names,
                    buffer_size=100
                )
                
                # Load historical data for replay
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(minutes=replay_minutes)
                
                df = db.get_ohlcv(replay_symbol, start_time, end_time)
                
                if df.empty:
                    st.error("No data available for replay.")
                    st.session_state.replay_running = False
                    st.stop()
                
                df = df.sort_values('time').reset_index(drop=True)
                
                total_candles = 0
                total_anomalies = 0
                recent_anomalies = []
                
                # Replay loop
                for idx, row in df.iterrows():
                    if not st.session_state.replay_running:
                        break
                    
                    start_process = time.time()
                    
                    candle = Candle(
                        time=row['time'].to_pydatetime() if hasattr(row['time'], 'to_pydatetime') else row['time'],
                        symbol=replay_symbol,
                        open=float(row['open']),
                        high=float(row['high']),
                        low=float(row['low']),
                        close=float(row['close']),
                        volume=float(row['volume'])
                    )
                    
                    prediction = inference.process_candle(candle)
                    
                    latency_ms = int((time.time() - start_process) * 1000)
                    
                    total_candles += 1
                    
                    # Store data for visualization
                    st.session_state.replay_data['times'].append(candle.time)
                    st.session_state.replay_data['prices'].append(candle.close)
                    
                    if prediction:
                        st.session_state.replay_data['scores'].append(prediction.anomaly_score)
                        st.session_state.replay_data['anomalies'].append(prediction.is_anomaly)
                        
                        if prediction.is_anomaly:
                            total_anomalies += 1
                            recent_anomalies.append({
                                'Time': candle.time,
                                'Price': candle.close,
                                'Score': prediction.anomaly_score
                            })
                            recent_anomalies = recent_anomalies[-10:]
                    else:
                        st.session_state.replay_data['scores'].append(0)
                        st.session_state.replay_data['anomalies'].append(False)
                    
                    # Update metrics
                    candles_placeholder.metric("Candles Processed", total_candles)
                    anomalies_placeholder.metric("Anomalies Detected", total_anomalies)
                    
                    if total_candles > 0:
                        rate = total_anomalies / total_candles
                        rate_placeholder.metric("Anomaly Rate", f"{rate:.2%}")
                    
                    latency_placeholder.metric("Latency", f"{latency_ms} ms")
                    
                    # Update chart periodically
                    if total_candles % 10 == 0:
                        times = list(st.session_state.replay_data['times'])
                        prices = list(st.session_state.replay_data['prices'])
                        anomalies = list(st.session_state.replay_data['anomalies'])
                        
                        if len(times) > 0:
                            fig = go.Figure()
                            
                            # Price line
                            fig.add_trace(go.Scatter(
                                x=times,
                                y=prices,
                                mode='lines',
                                name='Price',
                                line=dict(color='blue', width=1)
                            ))
                            
                            # Anomaly markers
                            anomaly_times = [t for t, a in zip(times, anomalies) if a]
                            anomaly_prices = [p for p, a in zip(prices, anomalies) if a]
                            
                            if anomaly_times:
                                fig.add_trace(go.Scatter(
                                    x=anomaly_times,
                                    y=anomaly_prices,
                                    mode='markers',
                                    name='Anomaly',
                                    marker=dict(color='red', size=10, symbol='x')
                                ))
                            
                            fig.update_layout(
                                title=f"{replay_symbol} Price with Anomaly Detection",
                                xaxis_title="Time",
                                yaxis_title="Price",
                                height=400
                            )
                            
                            chart_placeholder.plotly_chart(fig, use_container_width=True)
                        
                        # Update anomaly table
                        if recent_anomalies:
                            anomaly_table_placeholder.dataframe(
                                pd.DataFrame(recent_anomalies),
                                use_container_width=True
                            )
                    
                    # Control replay speed
                    time.sleep(1 / replay_speed)
                
                st.session_state.replay_running = False
                st.success(f"Replay complete. Processed {total_candles} candles, detected {total_anomalies} anomalies.")
                
            except Exception as e:
                st.session_state.replay_running = False
                st.error(f"Replay failed: {e}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())


with tab2:
    st.subheader("Live WebSocket Stream (Demo)")
    
    st.markdown("""
    This demonstrates the live streaming capability using Coinbase WebSocket feed.
    
    **Note:** This requires the streaming worker to be running:
    ```bash
    docker-compose --profile streaming up
    ```
    
    For the demo, we simulate the live feed using recent historical data.
    """)
    
    st.info("""
    **Production Usage:**
    
    In production, the streaming worker connects to the Coinbase WebSocket API
    and processes real-time trade data. Each trade is aggregated into candles,
    features are computed, and anomaly detection runs continuously.
    
    Predictions are:
    1. Published to Redis streams for real-time consumers
    2. Logged to TimescaleDB for historical analysis
    3. Displayed in this dashboard via polling
    """)
    
    # Architecture diagram
    st.markdown("#### Streaming Architecture")
    
    st.code("""
    Coinbase WebSocket
           |
           v
    Trade Aggregator (1-minute candles)
           |
           v
    Feature Computer (rolling windows via Redis cache)
           |
           v
    Model Inference (K-Means / GMM)
           |
           v
    +------------------+------------------+
    |                  |                  |
    v                  v                  v
Redis Stream    TimescaleDB       Streamlit UI
(real-time)     (history)         (monitoring)
    """, language="text")
    
    st.markdown("#### Recent Predictions")
    
    try:
        from src.data.database import db
        
        available_symbols = get_available_symbols()
        
        if available_symbols:
            stream_symbol = st.selectbox("Symbol", options=available_symbols, key="stream_symbol")
            
            # Get recent predictions
            predictions = db.get_predictions(
                symbol=stream_symbol,
                limit=100
            )
            
            if predictions.empty:
                st.info("No predictions logged yet. Run the streaming worker or historical replay to generate predictions.")
            else:
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Predictions", len(predictions))
                with col2:
                    anomaly_count = predictions['is_anomaly'].sum() if 'is_anomaly' in predictions.columns else 0
                    st.metric("Anomalies", int(anomaly_count))
                with col3:
                    if 'latency_ms' in predictions.columns:
                        avg_latency = predictions['latency_ms'].mean()
                        st.metric("Avg Latency", f"{avg_latency:.0f} ms")
                
                # Recent predictions table
                display_cols = [c for c in ['time', 'model_name', 'anomaly_score', 'is_anomaly', 'close_price', 'latency_ms'] if c in predictions.columns]
                st.dataframe(predictions[display_cols].head(20), use_container_width=True)
        else:
            st.warning("No symbols available. Download data first.")
            
    except Exception as e:
        st.warning(f"Could not connect to database: {e}")
