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
import threading
import queue
import asyncio

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
    """Load a model by run ID — tries local file first, then MLflow artifacts."""
    from config.settings import settings
    from src.models.base import BaseAnomalyDetector

    local_path = settings.data_dir / "models" / f"{run_id}.joblib"
    if local_path.exists():
        return BaseAnomalyDetector.load(local_path)

    import mlflow.sklearn
    return mlflow.sklearn.load_model(f"runs:/{run_id}/model")


model_options, run_id_map = get_mlflow_models()

# Tabs
tab1, tab2 = st.tabs(["Historical Replay", "Live Stream"])


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
                
                # Load most recent historical data for replay
                # First find the latest available timestamp, then work backwards
                status = db.get_data_status()
                symbol_status = status[status['symbol'] == replay_symbol]

                if symbol_status.empty:
                    st.error("No data available for replay.")
                    st.session_state.replay_running = False
                    st.stop()

                end_time = symbol_status.iloc[0]['latest'].to_pydatetime()
                start_time = end_time - timedelta(minutes=replay_minutes)

                df = db.get_ohlcv(replay_symbol, start_time, end_time)

                if df.empty:
                    st.error("No data available for replay.")
                    st.session_state.replay_running = False
                    st.stop()

                st.info(f"Replaying {len(df):,} candles from {start_time:%Y-%m-%d %H:%M} to {end_time:%Y-%m-%d %H:%M}")
                
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
    st.subheader("Live WebSocket Stream")

    st.markdown("""
    Connect to the Coinbase WebSocket feed for real-time anomaly detection.
    Trades are aggregated into candles at your chosen interval, features are
    computed on a rolling window, and the selected model scores each candle.
    """)

    if not model_options:
        st.warning("No trained models available. Train a model first.")
    else:
        # Coinbase symbols available for streaming
        COINBASE_SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "LINK-USD"]

        col1, col2, col3 = st.columns(3)

        with col1:
            live_model = st.selectbox("Select Model", options=model_options, key="live_model")

        with col2:
            live_symbols = st.multiselect(
                "Symbols",
                options=COINBASE_SYMBOLS,
                default=["BTC-USD"],
                key="live_symbols",
            )

        with col3:
            interval_option = st.select_slider(
                "Candle Interval",
                options=[30, 60, 120],
                value=60,
                format_func=lambda x: f"{x}s",
                key="live_interval",
            )

        # ---- session-state defaults ----
        if "live_running" not in st.session_state:
            st.session_state.live_running = False
        if "live_data" not in st.session_state:
            st.session_state.live_data = {
                "times": deque(maxlen=500),
                "prices": deque(maxlen=500),
                "symbols": deque(maxlen=500),
                "scores": deque(maxlen=500),
                "anomalies": deque(maxlen=500),
            }

        col_start, col_stop = st.columns(2)
        with col_start:
            start_live = st.button(
                "Start Live Stream",
                type="primary",
                disabled=st.session_state.live_running or len(live_symbols) == 0,
            )
        with col_stop:
            stop_live = st.button("Stop", disabled=not st.session_state.live_running)

        if stop_live:
            st.session_state.live_running = False

        # ---- metrics placeholders ----
        metric_cols = st.columns(4)
        with metric_cols[0]:
            live_candles_ph = st.empty()
        with metric_cols[1]:
            live_anomalies_ph = st.empty()
        with metric_cols[2]:
            live_rate_ph = st.empty()
        with metric_cols[3]:
            live_latency_ph = st.empty()

        # Connection status + buffer progress
        live_status_ph = st.empty()
        live_buffer_ph = st.empty()

        # Chart + table placeholders
        live_chart_ph = st.empty()
        live_table_ph = st.empty()

        # ---- background WebSocket thread ----
        def _ws_thread(symbols, interval, candle_queue, stop_event):
            """Run CoinbaseWebSocket in its own asyncio event loop."""
            from src.streaming.inference import CoinbaseWebSocket

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            ws = CoinbaseWebSocket(symbols, interval)

            def _on_candle(candle):
                if not stop_event.is_set():
                    candle_queue.put(candle)

            async def _run():
                task = asyncio.ensure_future(ws.stream(_on_candle))
                while not stop_event.is_set():
                    await asyncio.sleep(0.25)
                ws.stop()
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            try:
                loop.run_until_complete(_run())
            finally:
                loop.close()

        # ---- main streaming loop ----
        if start_live:
            st.session_state.live_running = True

            # Clear previous data
            for key in st.session_state.live_data:
                st.session_state.live_data[key].clear()

            try:
                from src.data.database import db
                from src.streaming.inference import StreamingInference

                # Load model
                run_id = run_id_map[live_model]
                model = load_model_from_mlflow(run_id)

                feature_names = model.feature_names or [
                    "volatility", "log_return", "volume_ratio",
                    "return_std", "price_range", "volatility_ratio",
                ]

                inference = StreamingInference(
                    model=model,
                    feature_names=feature_names,
                    buffer_size=100,
                )
                min_buffer = inference._min_buffer

                # Thread-safe primitives
                candle_queue: queue.Queue = queue.Queue()
                stop_event = threading.Event()

                thread = threading.Thread(
                    target=_ws_thread,
                    args=(live_symbols, interval_option, candle_queue, stop_event),
                    daemon=True,
                )
                thread.start()

                live_status_ph.success(
                    f"Connected — streaming {', '.join(live_symbols)} "
                    f"(interval {interval_option}s)"
                )

                total_candles = 0
                total_anomalies = 0
                total_predictions = 0
                recent_predictions: list[dict] = []

                model_type = live_model.split(" - ")[0] if " - " in live_model else "unknown"

                while st.session_state.live_running:
                    try:
                        candle = candle_queue.get(timeout=0.5)
                    except queue.Empty:
                        continue

                    start_process = time.time()
                    prediction = inference.process_candle(candle)
                    latency_ms = int((time.time() - start_process) * 1000)

                    total_candles += 1

                    # Show buffer fill progress per symbol
                    buf = inference._get_buffer(candle.symbol)
                    buf_len = len(buf)
                    if buf_len < min_buffer:
                        live_buffer_ph.warning(
                            f"Buffering {candle.symbol}: {buf_len}/{min_buffer} candles "
                            f"(need {min_buffer - buf_len} more before predictions start)"
                        )
                    elif total_predictions == 0:
                        live_buffer_ph.success("Buffer full — predictions starting.")
                    else:
                        live_buffer_ph.empty()

                    st.session_state.live_data["times"].append(candle.time)
                    st.session_state.live_data["prices"].append(candle.close)
                    st.session_state.live_data["symbols"].append(candle.symbol)

                    if prediction:
                        total_predictions += 1
                        st.session_state.live_data["scores"].append(prediction.anomaly_score)
                        st.session_state.live_data["anomalies"].append(prediction.is_anomaly)

                        if prediction.is_anomaly:
                            total_anomalies += 1

                        recent_predictions.append({
                            "Time": candle.time,
                            "Symbol": candle.symbol,
                            "Price": candle.close,
                            "Score": round(prediction.anomaly_score, 4),
                            "Anomaly": prediction.is_anomaly,
                        })
                        recent_predictions = recent_predictions[-20:]

                        # Log to TimescaleDB
                        try:
                            db.log_prediction(
                                time=candle.time,
                                symbol=candle.symbol,
                                model_name=model_type,
                                model_version=run_id[:8],
                                anomaly_score=prediction.anomaly_score,
                                is_anomaly=prediction.is_anomaly,
                                threshold=prediction.threshold,
                                close_price=candle.close,
                                latency_ms=latency_ms,
                                source="live_stream",
                            )
                        except Exception:
                            pass
                    else:
                        st.session_state.live_data["scores"].append(0)
                        st.session_state.live_data["anomalies"].append(False)

                    # Update metrics
                    live_candles_ph.metric("Candles Processed", total_candles)
                    live_anomalies_ph.metric("Anomalies Detected", total_anomalies)
                    if total_candles > 0:
                        live_rate_ph.metric("Anomaly Rate", f"{total_anomalies / total_candles:.2%}")
                    live_latency_ph.metric("Latency", f"{latency_ms} ms")

                    # Update chart every 5 candles
                    if total_candles % 5 == 0:
                        times = list(st.session_state.live_data["times"])
                        prices = list(st.session_state.live_data["prices"])
                        anomalies = list(st.session_state.live_data["anomalies"])

                        if times:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=times,
                                y=prices,
                                mode="lines",
                                name="Price",
                                line=dict(color="blue", width=1),
                            ))

                            anomaly_times = [t for t, a in zip(times, anomalies) if a]
                            anomaly_prices = [p for p, a in zip(prices, anomalies) if a]
                            if anomaly_times:
                                fig.add_trace(go.Scatter(
                                    x=anomaly_times,
                                    y=anomaly_prices,
                                    mode="markers",
                                    name="Anomaly",
                                    marker=dict(color="red", size=10, symbol="x"),
                                ))

                            symbols_label = ", ".join(sorted(set(st.session_state.live_data["symbols"])))
                            fig.update_layout(
                                title=f"Live Stream — {symbols_label}",
                                xaxis_title="Time",
                                yaxis_title="Price",
                                height=400,
                            )
                            live_chart_ph.plotly_chart(fig, use_container_width=True)

                    # Update predictions table on every prediction
                    if recent_predictions:
                        live_table_ph.dataframe(
                            pd.DataFrame(recent_predictions),
                            use_container_width=True,
                        )

                # Stream stopped — clean up
                stop_event.set()
                thread.join(timeout=5)
                live_status_ph.info("Stream stopped.")
                st.success(
                    f"Live stream ended. Processed {total_candles} candles, "
                    f"detected {total_anomalies} anomalies."
                )

            except Exception as e:
                st.session_state.live_running = False
                st.error(f"Live stream failed: {e}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())

        # ---- Recent predictions from DB (shown when not streaming) ----
        if not st.session_state.live_running:
            st.markdown("#### Recent Predictions")
            try:
                from src.data.database import db

                # Show predictions for any symbol that has been streamed
                from src.streaming.inference import CoinbaseWebSocket
                binance_symbols = [
                    CoinbaseWebSocket.SYMBOL_MAP.get(s, s) for s in live_symbols
                ]

                for sym in binance_symbols:
                    predictions = db.get_predictions(symbol=sym, limit=50)
                    if not predictions.empty:
                        st.markdown(f"**{sym}**")
                        display_cols = [
                            c for c in [
                                "time", "model_name", "anomaly_score",
                                "is_anomaly", "close_price", "latency_ms",
                            ] if c in predictions.columns
                        ]
                        st.dataframe(
                            predictions[display_cols].head(20),
                            use_container_width=True,
                        )
            except Exception:
                pass
