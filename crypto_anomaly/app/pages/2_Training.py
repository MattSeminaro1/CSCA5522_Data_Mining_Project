"""
Model Training Page.

Train and tune anomaly detection models with hyperparameter search.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Model Training", page_icon="chart_with_upwards_trend", layout="wide")

st.title("Model Training")
st.markdown("Train anomaly detection models with hyperparameter tuning.")


def load_training_data(symbols: list[str], days: int) -> pd.DataFrame:
    """Load and prepare training data from database."""
    from src.data.database import db
    
    all_data = []
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days)
    
    for symbol in symbols:
        df = db.get_ohlcv(symbol, start_time, end_time)
        if not df.empty:
            df['symbol'] = symbol
            all_data.append(df)
    
    if not all_data:
        return pd.DataFrame()
    
    return pd.concat(all_data, ignore_index=True)


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


# Tabs
tab1, tab2, tab3 = st.tabs(["Train Model", "Find Optimal K", "MLflow Experiments"])


with tab1:
    st.subheader("Train Anomaly Detection Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Model Configuration")
        model_name = st.text_input(
            "Model Name (optional)",
            placeholder="e.g., btc_volatility_detector",
            help="A descriptive name to identify this model. If empty, one will be auto-generated."
        )
                
        model_type = st.selectbox(
            "Model Type",
            options=["kmeans", "gmm"],
            format_func=lambda x: {"kmeans": "K-Means Clustering", "gmm": "Gaussian Mixture Model"}[x]
        )
        
        n_clusters = st.slider(
            "Number of Clusters/Components",
            min_value=2,
            max_value=20,
            value=5
        )
        
        contamination = st.slider(
            "Expected Anomaly Rate",
            min_value=0.01,
            max_value=0.20,
            value=0.05,
            step=0.01,
            help="Expected proportion of anomalies in the data"
        )
        
        covariance_type = "full"
        if model_type == "gmm":
            covariance_type = st.selectbox(
                "Covariance Type",
                options=["full", "tied", "diag", "spherical"],
                help="Type of covariance parameters to use"
            )
    
    with col2:
        st.markdown("#### Data Configuration")
        
        available_symbols = get_available_symbols()
        
        if not available_symbols:
            st.warning("No data available. Download data first in Data Management.")
            selected_symbols = []
        else:
            selected_symbols = st.multiselect(
                "Training Symbols",
                options=available_symbols,
                default=available_symbols[:min(3, len(available_symbols))]
            )
        
        from src.features.registry import feature_registry
        all_features = feature_registry.list_features()
        
        default_features = ["volatility", "log_return", "volume_ratio", "return_std", "price_range", "volatility_ratio"]
        default_features = [f for f in default_features if f in all_features]
        
        selected_features = st.multiselect(
            "Features",
            options=all_features,
            default=default_features
        )
        
        training_days = st.slider(
            "Training Days",
            min_value=30,
            max_value=365,
            value=180,
            help="Number of days of data to use for training"
        )
        
        test_split = st.slider(
            "Test Split",
            min_value=0.1,
            max_value=0.4,
            value=0.2,
            help="Proportion of data to use for testing"
        )
    
    st.markdown("---")
    
    can_train = len(selected_symbols) > 0 and len(selected_features) > 0
    
    if st.button("Train Model", type="primary", disabled=not can_train):
        
        with st.spinner("Loading and preparing data..."):
            try:
                from src.features.registry import compute_features, get_feature_matrix
                from src.models.trainer import ModelTrainer
                
                combined = load_training_data(selected_symbols, training_days)
                
                if combined.empty:
                    st.error("No data loaded. Check that selected symbols have data.")
                    st.stop()
                
                df_features = compute_features(combined, selected_features)
                X, df_clean = get_feature_matrix(df_features, selected_features)
                
                st.info(f"Training on {len(X):,} samples with {len(selected_features)} features")
                
            except Exception as e:
                st.error(f"Data preparation failed: {e}")
                st.stop()
        
        with st.spinner("Training model..."):
            try:
                split_idx = int(len(X) * (1 - test_split))
                X_train, X_test = X[:split_idx], X[split_idx:]
                
                if model_type == 'kmeans':
                    params = {
                        'n_clusters': n_clusters,
                        'contamination': contamination
                    }
                else:
                    params = {
                        'n_components': n_clusters,
                        'contamination': contamination,
                        'covariance_type': covariance_type
                    }
                if model_name.strip():
                    run_name = model_name.strip()
                else:
                    symbols_short = '_'.join([s[:3] for s in selected_symbols[:3]])
                    run_name = f"{model_type}_k{n_clusters}_{symbols_short}_cont{contamination}"
                    
                from config.settings import settings
                trainer = ModelTrainer(
                    mlflow_tracking_uri=settings.mlflow_tracking_uri,
                    use_mlflow=True
                )
                model, run_id = trainer.train(
                    model_type=model_type,
                    X_train=X_train,
                    feature_names=selected_features,
                    run_name=run_name,
                    params=params,
                    X_test=X_test,
                    tags={
                        'symbols': ','.join(selected_symbols),
                        'source': 'streamlit'
                    },
                    register_model=True
                )
                
                st.success("Model trained successfully!")
                
                col1, col2, col3, col4 = st.columns(4)
                
                model_params = model.get_model_params()
                
                with col1:
                    sil_score = model_params.get('silhouette_score')
                    st.metric("Silhouette Score", f"{sil_score:.4f}" if sil_score else "N/A")
                
                with col2:
                    st.metric("Threshold", f"{model.threshold:.4f}")
                
                with col3:
                    train_preds = model.predict(X_train)
                    train_rate = train_preds.mean()
                    st.metric("Train Anomaly Rate", f"{train_rate:.2%}")
                
                with col4:
                    test_preds = model.predict(X_test)
                    test_rate = test_preds.mean()
                    st.metric("Test Anomaly Rate", f"{test_rate:.2%}")
                
                st.markdown("#### Anomaly Score Distribution")
                
                train_scores = model.score_samples(X_train)
                test_scores = model.score_samples(X_test)
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=train_scores, 
                    name="Train", 
                    opacity=0.7,
                    nbinsx=50
                ))
                fig.add_trace(go.Histogram(
                    x=test_scores, 
                    name="Test", 
                    opacity=0.7,
                    nbinsx=50
                ))
                fig.add_vline(
                    x=model.threshold, 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text="Threshold"
                )
                fig.update_layout(
                    barmode='overlay', 
                    title="Anomaly Score Distribution",
                    xaxis_title="Anomaly Score",
                    yaxis_title="Count"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                if run_id:
                    st.info(f"MLflow Run ID: `{run_id}`")
                
            except Exception as e:
                st.error(f"Training failed: {e}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())


with tab2:
    st.subheader("Find Optimal Number of Clusters")
    
    st.markdown("Use the elbow method and silhouette analysis to find the optimal number of clusters.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        opt_model_type = st.selectbox(
            "Model Type",
            options=["kmeans", "gmm"],
            key="opt_model",
            format_func=lambda x: {"kmeans": "K-Means", "gmm": "GMM"}[x]
        )
        
        k_min, k_max = st.slider(
            "K Range",
            min_value=2,
            max_value=20,
            value=(2, 12),
            key="k_range"
        )
    
    with col2:
        available_symbols = get_available_symbols()
        
        opt_selected_symbols = st.multiselect(
            "Symbols for Analysis",
            options=available_symbols,
            default=available_symbols[:min(2, len(available_symbols))] if available_symbols else [],
            key="opt_symbols"
        )
        
        opt_days = st.slider(
            "Days of Data",
            min_value=30,
            max_value=180,
            value=90,
            key="opt_days"
        )
    
    can_analyze = len(opt_selected_symbols) > 0
    
    if st.button("Find Optimal K", disabled=not can_analyze):
        with st.spinner("Analyzing..."):
            try:
                from src.features.registry import compute_features, get_feature_matrix
                from src.models.trainer import find_best_k
                
                combined = load_training_data(opt_selected_symbols, opt_days)
                
                if combined.empty:
                    st.error("No data loaded.")
                    st.stop()
                
                default_features = ["volatility", "log_return", "volume_ratio", "return_std", "price_range", "volatility_ratio"]
                df_features = compute_features(combined, default_features)
                X, _ = get_feature_matrix(df_features, default_features)
                
                max_samples = 50000
                if len(X) > max_samples:
                    idx = np.random.choice(len(X), max_samples, replace=False)
                    X = X[idx]
                
                st.info(f"Analyzing {len(X):,} samples...")
                
                k_range = range(k_min, k_max + 1)
                results = find_best_k(X, opt_model_type, k_range)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if opt_model_type == 'kmeans':
                        fig = px.line(
                            x=results['k'],
                            y=results['inertia'],
                            title="Elbow Method (Inertia)",
                            labels={'x': 'Number of Clusters (K)', 'y': 'Inertia'},
                            markers=True
                        )
                        if 'elbow_k' in results:
                            fig.add_vline(
                                x=results['elbow_k'], 
                                line_dash="dash",
                                annotation_text=f"Elbow: {results['elbow_k']}"
                            )
                    else:
                        fig = px.line(
                            x=results['n'],
                            y=results['bic'],
                            title="BIC Score (Lower is Better)",
                            labels={'x': 'Number of Components', 'y': 'BIC'},
                            markers=True
                        )
                        if 'best_bic_n' in results:
                            fig.add_vline(
                                x=results['best_bic_n'], 
                                line_dash="dash",
                                annotation_text=f"Best: {results['best_bic_n']}"
                            )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    k_values = results.get('k', results.get('n'))
                    fig = px.line(
                        x=k_values,
                        y=results['silhouette'],
                        title="Silhouette Score (Higher is Better)",
                        labels={'x': 'Number of Clusters', 'y': 'Silhouette Score'},
                        markers=True
                    )
                    
                    best_sil_key = 'best_silhouette_k' if opt_model_type == 'kmeans' else 'best_silhouette_n'
                    if best_sil_key in results:
                        fig.add_vline(
                            x=results[best_sil_key], 
                            line_dash="dash",
                            annotation_text=f"Best: {results[best_sil_key]}"
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("#### Recommendations")
                
                if opt_model_type == 'kmeans':
                    elbow_k = results.get('elbow_k', 'N/A')
                    best_sil_k = results.get('best_silhouette_k', 'N/A')
                    st.info(f"Elbow Method suggests K = {elbow_k}. Best Silhouette at K = {best_sil_k}.")
                else:
                    best_bic = results.get('best_bic_n', 'N/A')
                    best_sil = results.get('best_silhouette_n', 'N/A')
                    st.info(f"Best BIC at N = {best_bic}. Best Silhouette at N = {best_sil}.")
                    
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())


with tab3:
    st.subheader("MLflow Experiment Tracking")
    
    try:
        import mlflow
        from config.settings import settings
        
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        
        st.markdown(f"**MLflow Tracking URI:** `{settings.mlflow_external_uri}`")
        
        try:
            experiments = mlflow.search_experiments()
            
            if experiments:
                exp_names = [exp.name for exp in experiments if exp.name != "Default"]
                
                if exp_names:
                    selected_exp = st.selectbox("Select Experiment", options=exp_names)
                    
                    if selected_exp:
                        exp = next((e for e in experiments if e.name == selected_exp), None)
                        
                        if exp:
                            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
                            
                            if not runs.empty:
                                st.markdown(f"**{len(runs)} runs found**")
                                
                                # Add run_name column from MLflow tags
                                if 'tags.mlflow.runName' in runs.columns:
                                    runs.insert(0, 'run_name', runs['tags.mlflow.runName'])

                                display_cols = ['run_name', 'run_id', 'status', 'start_time']
                                param_cols = [c for c in runs.columns if c.startswith('params.')]
                                metric_cols = [c for c in runs.columns if c.startswith('metrics.')]

                                display_cols.extend(param_cols[:5])
                                display_cols.extend(metric_cols[:5])
                                display_cols = [c for c in display_cols if c in runs.columns]

                                st.dataframe(runs[display_cols], use_container_width=True)
                            else:
                                st.info("No runs found in this experiment.")
                else:
                    st.info("No experiments found. Train a model to create experiments.")
            else:
                st.info("No experiments found. Train a model to create experiments.")
                
        except Exception as e:
            st.warning(f"Could not list experiments: {e}")
        
        st.markdown("---")
        st.markdown(f"[Open MLflow UI]({settings.mlflow_external_uri})")
        
    except ImportError:
        st.warning("MLflow not installed.")
    except Exception as e:
        st.warning(f"Could not connect to MLflow: {e}")
