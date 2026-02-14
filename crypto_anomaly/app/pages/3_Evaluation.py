"""
Model Evaluation Page.

Compare model performance and analyze results.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Model Evaluation", page_icon="chart_with_upwards_trend", layout="wide")

st.title("Model Evaluation")
st.markdown("Evaluate and compare anomaly detection model performance.")


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


def load_model_from_mlflow(run_id: str):
    """Load a model by run ID — tries local file first, then MLflow artifacts."""
    from config.settings import settings
    from src.models.base import BaseAnomalyDetector

    # Try local model file first
    local_path = settings.data_dir / "models" / f"{run_id}.joblib"
    if local_path.exists():
        return BaseAnomalyDetector.load(local_path)

    # Fall back to MLflow artifact store
    import mlflow.sklearn
    return mlflow.sklearn.load_model(f"runs:/{run_id}/model")


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
                    run_name = run.get('tags.mlflow.runName', '')
                    label = f"{model_type}-{run_name}-{run_id[:8]}" if run_name else f"{model_type}-{run_id[:8]}"
                    model_options.append(label)
                    run_id_map[label] = run_id
    except Exception:
        pass
    
    return model_options, run_id_map


model_options, run_id_map = get_mlflow_models()

# Tabs
tab1, tab2, tab3 = st.tabs(["Evaluate Model", "Compare Models", "Anomaly Analysis"])


with tab1:
    st.subheader("Evaluate Single Model")
    
    if not model_options:
        st.info("No trained models found. Train a model first in the Training page.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            selected_model = st.selectbox("Select Model", options=model_options)
            
            available_symbols = get_available_symbols()
            eval_symbols = st.multiselect(
                "Evaluation Symbols",
                options=available_symbols,
                default=available_symbols[:min(2, len(available_symbols))] if available_symbols else []
            )
        
        with col2:
            eval_days = st.slider(
                "Evaluation Days",
                min_value=7,
                max_value=90,
                value=30
            )
        
        if st.button("Evaluate", type="primary", disabled=len(eval_symbols) == 0):
            with st.spinner("Evaluating model..."):
                try:
                    from src.data.database import db
                    from src.features.registry import compute_features, get_feature_matrix
                    
                    run_id = run_id_map[selected_model]
                    model = load_model_from_mlflow(run_id)
                    
                    all_data = []
                    end_time = datetime.utcnow()
                    start_time = end_time - timedelta(days=eval_days)
                    
                    for symbol in eval_symbols:
                        df = db.get_ohlcv(symbol, start_time, end_time)
                        if not df.empty:
                            df['symbol'] = symbol
                            all_data.append(df)
                    
                    if not all_data:
                        st.error("No data found for evaluation period.")
                        st.stop()
                    
                    combined = pd.concat(all_data, ignore_index=True)
                    
                    feature_names = model.feature_names or [
                        "volatility", "log_return", "volume_ratio",
                        "return_std", "price_range", "volatility_ratio"
                    ]
                    
                    df_features = compute_features(combined, feature_names)
                    X, df_clean = get_feature_matrix(df_features, feature_names)
                    
                    scores = model.score_samples(X)
                    predictions = model.predict(X)
                    
                    st.markdown("#### Evaluation Results")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Samples", f"{len(X):,}")
                    with col2:
                        st.metric("Anomalies Detected", f"{int(predictions.sum()):,}")
                    with col3:
                        st.metric("Anomaly Rate", f"{predictions.mean():.2%}")
                    with col4:
                        st.metric("Avg Score", f"{scores.mean():.4f}")
                    
                    st.markdown("#### Score Distribution")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x=scores, nbinsx=50, name="Anomaly Scores"))
                    fig.add_vline(
                        x=model.threshold,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Threshold: {model.threshold:.4f}"
                    )
                    fig.update_layout(
                        title="Anomaly Score Distribution",
                        xaxis_title="Score",
                        yaxis_title="Count"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if 'time' in df_clean.columns:
                        st.markdown("#### Anomalies Over Time")
                        
                        df_results = df_clean.copy()
                        df_results['anomaly_score'] = scores
                        df_results['is_anomaly'] = predictions
                        df_results['date'] = pd.to_datetime(df_results['time']).dt.date
                        
                        daily_stats = df_results.groupby('date').agg({
                            'is_anomaly': 'sum',
                            'anomaly_score': 'mean'
                        }).reset_index()
                        
                        fig = make_subplots(specs=[[{"secondary_y": True}]])
                        
                        fig.add_trace(
                            go.Bar(x=daily_stats['date'], y=daily_stats['is_anomaly'], name="Anomaly Count"),
                            secondary_y=False
                        )
                        fig.add_trace(
                            go.Scatter(x=daily_stats['date'], y=daily_stats['anomaly_score'], name="Avg Score", mode='lines'),
                            secondary_y=True
                        )
                        
                        fig.update_layout(title="Daily Anomaly Statistics")
                        fig.update_yaxes(title_text="Anomaly Count", secondary_y=False)
                        fig.update_yaxes(title_text="Avg Score", secondary_y=True)
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("#### Top Anomalies")
                    
                    df_results = df_clean.copy()
                    df_results['anomaly_score'] = scores
                    df_results['is_anomaly'] = predictions
                    
                    top_anomalies = df_results[df_results['is_anomaly'] == 1].nlargest(20, 'anomaly_score')
                    
                    if not top_anomalies.empty:
                        display_cols = [c for c in ['time', 'symbol', 'close', 'anomaly_score'] if c in top_anomalies.columns]
                        st.dataframe(top_anomalies[display_cols], use_container_width=True)
                    else:
                        st.info("No anomalies detected in evaluation period.")
                    
                except Exception as e:
                    st.error(f"Evaluation failed: {e}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())


with tab2:
    st.subheader("Compare Multiple Models")
    
    if len(model_options) < 2:
        st.info("Need at least 2 trained models to compare.")
    else:
        selected_models = st.multiselect(
            "Select Models to Compare",
            options=model_options,
            default=model_options[:min(2, len(model_options))]
        )
        
        if len(selected_models) >= 2:
            available_symbols = get_available_symbols()
            
            compare_symbols = st.multiselect(
                "Comparison Symbols",
                options=available_symbols,
                default=available_symbols[:1] if available_symbols else [],
                key="compare_symbols"
            )
            
            compare_days = st.slider("Comparison Days", min_value=7, max_value=60, value=14, key="compare_days")
            
            if st.button("Compare Models", type="primary", disabled=len(compare_symbols) == 0):
                with st.spinner("Comparing models..."):
                    try:
                        from src.data.database import db
                        from src.features.registry import compute_features, get_feature_matrix
                        
                        all_data = []
                        end_time = datetime.utcnow()
                        start_time = end_time - timedelta(days=compare_days)
                        
                        for symbol in compare_symbols:
                            df = db.get_ohlcv(symbol, start_time, end_time)
                            if not df.empty:
                                df['symbol'] = symbol
                                all_data.append(df)
                        
                        if not all_data:
                            st.error("No data found.")
                            st.stop()
                        
                        combined = pd.concat(all_data, ignore_index=True)
                        
                        comparison_results = []
                        
                        for model_label in selected_models:
                            run_id = run_id_map[model_label]
                            model = load_model_from_mlflow(run_id)
                            
                            feature_names = model.feature_names or [
                                "volatility", "log_return", "volume_ratio",
                                "return_std", "price_range", "volatility_ratio"
                            ]
                            
                            df_features = compute_features(combined.copy(), feature_names)
                            X, _ = get_feature_matrix(df_features, feature_names)
                            
                            scores = model.score_samples(X)
                            predictions = model.predict(X)
                            model_params = model.get_model_params()
                            
                            comparison_results.append({
                                'Model': model_label[:30],
                                'Silhouette': model_params.get('silhouette_score', 0),
                                'Threshold': model.threshold,
                                'Anomalies': int(predictions.sum()),
                                'Anomaly Rate': predictions.mean(),
                                'Avg Score': scores.mean()
                            })
                        
                        comparison_df = pd.DataFrame(comparison_results)
                        st.dataframe(comparison_df, use_container_width=True)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig = px.bar(comparison_df, x='Model', y='Silhouette', title="Silhouette Score")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            fig = px.bar(comparison_df, x='Model', y='Anomaly Rate', title="Anomaly Rate")
                            st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Comparison failed: {e}")


with tab3:
    st.subheader("Anomaly Analysis (Live Detection History)")

    st.markdown("Browse anomalies detected during live or historical replay sessions from the **Live Detection** page.")
    
    try:
        from src.data.database import db
        
        available_symbols = get_available_symbols()
        
        if not available_symbols:
            st.warning("No data available.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                analysis_symbol = st.selectbox("Symbol", options=available_symbols, key="analysis_symbol")
            with col2:
                analysis_days = st.slider("Days to Analyze", min_value=1, max_value=30, value=7, key="analysis_days")
            
            try:
                anomalies = db.get_predictions(symbol=analysis_symbol, anomalies_only=True, limit=500)
                
                if anomalies.empty:
                    st.info("No anomalies found in prediction history.")
                else:
                    st.markdown(f"**{len(anomalies)} anomalies found**")
                    
                    if 'time' in anomalies.columns:
                        fig = px.scatter(
                            anomalies, x='time', y='anomaly_score',
                            color='model_name' if 'model_name' in anomalies.columns else None,
                            size='anomaly_score',
                            title="Detected Anomalies Over Time"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    fig = px.histogram(anomalies, x='anomaly_score', nbins=30, title="Anomaly Score Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("#### Recent Anomalies")
                    display_cols = [c for c in ['time', 'model_name', 'anomaly_score', 'threshold_used', 'close_price'] if c in anomalies.columns]
                    st.dataframe(anomalies[display_cols].head(50), use_container_width=True)
                    
            except Exception as e:
                st.warning(f"Could not load prediction history: {e}")
                
    except Exception as e:
        st.error(f"Analysis failed: {e}")
