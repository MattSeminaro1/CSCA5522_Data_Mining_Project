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


def load_training_data(symbols: list[str], start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Load and prepare training data from database."""
    from src.data.database import db

    all_data = []

    for symbol in symbols:
        df = db.get_ohlcv(symbol, start_date, end_date)
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
tab1, tab2, tab3 = st.tabs(["Train Model", "Hyperparameter Tuning", "MLflow Experiments"])


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

        with st.expander("Advanced Parameters"):
            n_init = st.slider(
                "Number of Initializations (n_init)",
                min_value=1,
                max_value=50,
                value=10 if model_type == "kmeans" else 5,
                help="Number of times the algorithm runs with different seeds. Best result is kept."
            )

            max_iter = st.slider(
                "Max Iterations",
                min_value=50,
                max_value=1000,
                value=300 if model_type == "kmeans" else 200,
                step=50,
                help="Maximum iterations per run. Increase if the model does not converge."
            )

            random_state = st.number_input(
                "Random Seed",
                min_value=0,
                max_value=99999,
                value=42,
                help="Seed for reproducibility. Same seed produces identical results."
            )

            scale_features = st.checkbox(
                "Scale Features (StandardScaler)",
                value=True,
                help="Standardize features before clustering. Recommended unless features are already on the same scale."
            )

            if model_type == "kmeans":
                init_method = st.selectbox(
                    "Initialization Method",
                    options=["k-means++", "random"],
                    help="k-means++ selects initial centroids intelligently. random picks random data points."
                )
                algorithm = st.selectbox(
                    "Algorithm",
                    options=["lloyd", "elkan"],
                    help="lloyd is the standard algorithm. elkan can be faster for well-separated clusters."
                )
            else:
                init_params = st.selectbox(
                    "Initialization Method",
                    options=["kmeans", "k-means++", "random", "random_from_data"],
                    help="Method for initializing GMM weights, means, and covariances."
                )
                reg_covar = st.select_slider(
                    "Covariance Regularization (reg_covar)",
                    options=[1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3],
                    value=1e-6,
                    format_func=lambda x: f"{x:.0e}",
                    help="Regularization on covariance diagonal. Increase if you get convergence errors."
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
        
        date_col1, date_col2 = st.columns(2)
        with date_col1:
            train_start_date = st.date_input(
                "Start Date",
                value=datetime.utcnow().date() - timedelta(days=180),
                key="train_start_date",
            )
        with date_col2:
            train_end_date = st.date_input(
                "End Date",
                value=datetime.utcnow().date(),
                key="train_end_date",
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
                
                combined = load_training_data(
                    selected_symbols,
                    datetime.combine(train_start_date, datetime.min.time()),
                    datetime.combine(train_end_date, datetime.max.time()),
                )
                
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
                        'contamination': contamination,
                        'n_init': n_init,
                        'max_iter': max_iter,
                        'random_state': random_state,
                        'scale_features': scale_features,
                        'init': init_method,
                        'algorithm': algorithm,
                    }
                else:
                    params = {
                        'n_components': n_clusters,
                        'contamination': contamination,
                        'covariance_type': covariance_type,
                        'n_init': n_init,
                        'max_iter': max_iter,
                        'random_state': random_state,
                        'scale_features': scale_features,
                        'init_params': init_params,
                        'reg_covar': reg_covar,
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

                # Feature Analysis
                st.markdown("#### Feature Analysis")

                train_preds_fa = model.predict(X_train)
                anomaly_mask = train_preds_fa == 1

                if anomaly_mask.any() and (~anomaly_mask).any():
                    # Scale data for fair cross-feature comparison
                    if model.scaler is not None:
                        X_scaled = model.scaler.transform(X_train)
                    else:
                        X_scaled = X_train

                    # Method A: Anomaly vs normal difference
                    normal_mean = np.abs(X_scaled[~anomaly_mask]).mean(axis=0)
                    anomaly_mean = np.abs(X_scaled[anomaly_mask]).mean(axis=0)
                    importance = np.abs(anomaly_mean - normal_mean)
                    if importance.max() > 0:
                        importance = importance / importance.max()

                    # Method B: Cluster center spread (in scaled space)
                    if hasattr(model, 'get_cluster_centers'):
                        centers_scaled = model.model.cluster_centers_
                    else:
                        centers_scaled = model.model.means_
                    center_spread = np.std(centers_scaled, axis=0)
                    if center_spread.max() > 0:
                        center_spread_norm = center_spread / center_spread.max()
                    else:
                        center_spread_norm = center_spread

                    fa_col1, fa_col2 = st.columns(2)

                    with fa_col1:
                        importance_df = pd.DataFrame({
                            'Feature': selected_features,
                            'Anomaly Contribution': importance,
                            'Cluster Spread': center_spread_norm
                        }).sort_values('Anomaly Contribution', ascending=True)

                        fig_imp = px.bar(
                            importance_df,
                            y='Feature',
                            x='Anomaly Contribution',
                            orientation='h',
                            title='Feature Importance (Anomaly Contribution)',
                            color='Anomaly Contribution',
                            color_continuous_scale='Reds'
                        )
                        fig_imp.update_layout(yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig_imp, use_container_width=True)

                    with fa_col2:
                        if hasattr(model, 'get_cluster_centers'):
                            centers = model.get_cluster_centers()
                            center_label = "Cluster"
                        else:
                            centers = model.get_component_means()
                            center_label = "Component"

                        centers_df = pd.DataFrame(
                            centers,
                            columns=selected_features,
                            index=[f"{center_label} {i}" for i in range(len(centers))]
                        )

                        fig_heat = px.imshow(
                            centers_df,
                            title=f'{center_label} Centers by Feature',
                            labels=dict(x="Feature", y=center_label, color="Value"),
                            aspect="auto",
                            color_continuous_scale='RdBu_r'
                        )
                        st.plotly_chart(fig_heat, use_container_width=True)
                else:
                    st.info("Feature analysis requires both normal and anomalous samples.")

                if run_id:
                    st.info(f"MLflow Run ID: `{run_id}`")
                
            except Exception as e:
                st.error(f"Training failed: {e}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())


with tab2:
    st.subheader("Hyperparameter Tuning")
    st.markdown("Find the best model parameters automatically or configure a custom search.")

    # ── Auto-Tune (one-click) ───────────────────────────────────────
    st.markdown("#### Auto-Tune")
    st.markdown(
        "Select a model type and your data, then press the button. "
        "A curated grid search runs automatically and recommends the best parameters."
    )

    auto_col1, auto_col2 = st.columns(2)

    with auto_col1:
        auto_model_type = st.selectbox(
            "Model Type",
            options=["kmeans", "gmm"],
            key="auto_model_type",
            format_func=lambda x: {"kmeans": "K-Means Clustering", "gmm": "Gaussian Mixture Model"}[x]
        )

        available_symbols_auto = get_available_symbols()
        auto_symbols = st.multiselect(
            "Training Symbols",
            options=available_symbols_auto,
            default=available_symbols_auto[:min(3, len(available_symbols_auto))] if available_symbols_auto else [],
            key="auto_symbols"
        )

    with auto_col2:
        from src.features.registry import feature_registry as _auto_fr
        _auto_all_features = _auto_fr.list_features()
        _auto_default = ["volatility", "log_return", "volume_ratio", "return_std", "price_range", "volatility_ratio"]
        _auto_default = [f for f in _auto_default if f in _auto_all_features]

        auto_features = st.multiselect(
            "Features",
            options=_auto_all_features,
            default=_auto_default,
            key="auto_features"
        )

        auto_date_col1, auto_date_col2 = st.columns(2)
        with auto_date_col1:
            auto_start_date = st.date_input(
                "Start Date",
                value=datetime.utcnow().date() - timedelta(days=180),
                key="auto_start_date",
            )
        with auto_date_col2:
            auto_end_date = st.date_input(
                "End Date",
                value=datetime.utcnow().date(),
                key="auto_end_date",
            )
        auto_test_split = st.slider("Test Split", min_value=0.1, max_value=0.4, value=0.2, key="auto_test_split")

    can_auto = len(auto_symbols) > 0 and len(auto_features) > 0

    if auto_model_type == "kmeans":
        auto_grid = {
            'n_clusters': [3, 4, 5, 6, 7, 8, 10],
            'n_init': [3],  # Reduced from 10 for memory efficiency during tuning
        }
        auto_metric = "silhouette_score"
        n_auto = 7
    else:
        auto_grid = {
            'n_components': [3, 4, 5, 6, 7, 8, 10],
            'covariance_type': ['full', 'diag'],
            'n_init': [3],  # Reduced from 5 for memory efficiency during tuning
        }
        auto_metric = "silhouette_score"
        n_auto = 7 * 2  # 14 combinations

    auto_cluster_label = "n_clusters" if auto_model_type == "kmeans" else "n_components"

    # Feature subset search
    auto_search_features = st.checkbox(
        "Search over feature combinations",
        value=True,
        key="auto_search_features",
        help="Try different subsets of the selected features to find the best combination"
    )

    auto_feature_subsets = None
    n_feature_combos = 1
    if auto_search_features and len(auto_features) >= 3:
        auto_min_subset = st.slider(
            "Minimum subset size",
            min_value=2,
            max_value=max(2, len(auto_features) - 1),
            value=min(3, max(2, len(auto_features) - 1)),
            key="auto_min_subset"
        )

        from itertools import combinations as _combinations
        auto_feature_subsets = []
        for size in range(auto_min_subset, len(auto_features) + 1):
            auto_feature_subsets.extend([list(c) for c in _combinations(auto_features, size)])

        n_feature_combos = len(auto_feature_subsets)
        total_auto = n_auto * n_feature_combos

        st.caption(f"{n_feature_combos} feature subsets x {n_auto} param combos = **{total_auto}** total evaluations")

        if total_auto > 500:
            st.warning(f"Large search space ({total_auto} total). Consider increasing minimum subset size.")
    elif auto_search_features:
        st.info("Select at least 3 features to enable feature combination search.")

    with st.expander("Preview auto-tune grid", expanded=False):
        st.json({k: str(v) for k, v in auto_grid.items()})
        total_display = n_auto * n_feature_combos
        st.caption(f"{total_display} total combinations will be evaluated, optimizing **{auto_metric}**.")

    if st.button("Find Best Parameters", type="primary", disabled=not can_auto, key="auto_tune"):
        with st.spinner("Loading and preparing data..."):
            try:
                from src.features.registry import compute_features, get_feature_matrix
                from src.models.trainer import ModelTrainer

                combined = load_training_data(
                    auto_symbols,
                    datetime.combine(auto_start_date, datetime.min.time()),
                    datetime.combine(auto_end_date, datetime.max.time()),
                )
                if combined.empty:
                    st.error("No data loaded. Check that selected symbols have data.")
                    st.stop()

                df_features = compute_features(combined, auto_features)
                X, df_clean = get_feature_matrix(df_features, auto_features)

                split_idx = int(len(X) * (1 - auto_test_split))
                X_train, X_test = X[:split_idx], X[split_idx:]

                st.info(f"Data prepared: {len(X_train):,} train / {len(X_test):,} test samples")
            except Exception as e:
                st.error(f"Data preparation failed: {e}")
                st.stop()

        auto_total = n_auto * n_feature_combos
        auto_progress_bar = st.progress(0, text=f"0/{auto_total} combinations completed")

        try:
            from config.settings import settings
            trainer = ModelTrainer(
                mlflow_tracking_uri=settings.mlflow_tracking_uri,
                use_mlflow=True
            )

            def _auto_progress(current, total, result):
                pct = current / total
                score_text = ""
                if result and auto_metric in result:
                    score_text = f" — latest {auto_metric}: {result[auto_metric]:.4f}"
                auto_progress_bar.progress(pct, text=f"{current}/{total} combinations completed{score_text}")

            tuning_results = trainer.train_with_tuning(
                model_type=auto_model_type,
                X_train=X_train,
                feature_names=auto_features,
                param_grid=auto_grid,
                feature_subsets=auto_feature_subsets,
                X_test=X_test,
                metric=auto_metric,
                tags={
                    'symbols': ','.join(auto_symbols),
                    'source': 'streamlit_auto_tune'
                },
                on_progress=_auto_progress,
            )

            best_model = tuning_results['best_model']
            best_score = tuning_results['best_score']
            best_run_id = tuning_results['best_run_id']
            all_results = tuning_results['all_results']

            if best_model is None:
                st.warning("Auto-tune completed but no valid results were found.")
                st.stop()

            st.success(f"Auto-tune complete! Best silhouette score: **{best_score:.4f}**")

            # ── Recommended Parameters ──
            st.markdown("#### Recommended Parameters")
            st.markdown("Use these values in the **Train Model** tab to train your final model.")

            best_result = next((r for r in all_results if r['run_id'] == best_run_id), None)
            best_params = best_result['params'] if best_result else best_model.get_model_params()

            # Display as clean key-value metrics
            param_cols = st.columns(min(len(best_params), 4))
            for i, (key, value) in enumerate(best_params.items()):
                with param_cols[i % len(param_cols)]:
                    display_val = f"{value:.0e}" if isinstance(value, float) and value < 0.001 else str(value)
                    st.metric(key, display_val)

            # Copyable code block
            param_str = ",\n    ".join(f'"{k}": {repr(v)}' for k, v in best_params.items())
            st.code(f"{{\n    {param_str}\n}}", language="python")

            # Show best features if feature subset search was used
            best_features = tuning_results.get('best_features')
            if best_features and auto_feature_subsets:
                st.markdown(f"**Best Features** ({len(best_features)}): `{', '.join(best_features)}`")

            # ── Performance Summary ──
            st.markdown("#### Best Model Performance")
            model_params = best_model.get_model_params()
            perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)

            # Slice to best feature subset if needed
            if best_features and best_features != auto_features:
                best_col_idx = [auto_features.index(f) for f in best_features]
                X_train_best = X_train[:, best_col_idx]
                X_test_best = X_test[:, best_col_idx]
            else:
                X_train_best = X_train
                X_test_best = X_test

            with perf_col1:
                sil = model_params.get('silhouette_score')
                st.metric("Silhouette Score", f"{sil:.4f}" if sil else "N/A")
            with perf_col2:
                st.metric("Threshold", f"{best_model.threshold:.4f}")
            with perf_col3:
                train_preds = best_model.predict(X_train_best)
                st.metric("Train Anomaly Rate", f"{train_preds.mean():.2%}")
            with perf_col4:
                test_preds = best_model.predict(X_test_best)
                st.metric("Test Anomaly Rate", f"{test_preds.mean():.2%}")

            if best_run_id:
                st.info(f"Best MLflow Run ID: `{best_run_id}`")

            # ── Visualization ──
            with st.expander("Detailed Results", expanded=False):
                results_rows = []
                for r in all_results:
                    row = {**r['params']}
                    row[auto_metric] = r.get(auto_metric)
                    row['threshold'] = r.get('threshold')
                    if 'features' in r:
                        row['n_features'] = r.get('n_features')
                        row['features'] = ', '.join(r.get('features', []))
                    row['run_id'] = r.get('run_id', '')[:8] if r.get('run_id') else ''
                    results_rows.append(row)

                results_df = pd.DataFrame(results_rows)
                if auto_metric in results_df.columns:
                    results_df = results_df.sort_values(auto_metric, ascending=False)

                st.markdown("##### All Results")
                st.dataframe(results_df, use_container_width=True)

                st.markdown("##### Results Visualization")
                if auto_metric in results_df.columns and auto_cluster_label in results_df.columns:
                    viz_col1, viz_col2 = st.columns(2)

                    with viz_col1:
                        fig = px.scatter(
                            results_df,
                            x=auto_cluster_label,
                            y=auto_metric,
                            color='contamination' if 'contamination' in results_df.columns else None,
                            title=f"{auto_metric} vs {auto_cluster_label}",
                            hover_data=results_df.columns.tolist()
                        )
                        if best_result:
                            fig.add_trace(go.Scatter(
                                x=[best_result['params'].get(auto_cluster_label)],
                                y=[best_result.get(auto_metric)],
                                mode='markers',
                                marker=dict(size=15, symbol='star', color='red', line=dict(width=2, color='black')),
                                name='Best',
                                showlegend=True
                            ))
                        st.plotly_chart(fig, use_container_width=True)

                    with viz_col2:
                        fig = px.box(
                            results_df,
                            x=auto_cluster_label,
                            y=auto_metric,
                            title=f"{auto_metric} Distribution by {auto_cluster_label}"
                        )
                        st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Auto-tune failed: {e}")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())

    st.markdown("---")

    # ── Custom Grid Search ──────────────────────────────────────────
    with st.expander("Custom Grid Search", expanded=False):
        st.markdown("Build your own parameter grid for full control over the search space.")

        tune_col1, tune_col2 = st.columns(2)

        with tune_col1:
            st.markdown("##### Model & Metric")

            tune_model_type = st.selectbox(
                "Model Type",
                options=["kmeans", "gmm"],
                key="tune_model_type",
                format_func=lambda x: {"kmeans": "K-Means Clustering", "gmm": "Gaussian Mixture Model"}[x]
            )

            if tune_model_type == "gmm":
                metric_options = ["silhouette_score", "bic"]
                metric_help = "silhouette_score: higher is better. bic: lower is better."
            else:
                metric_options = ["silhouette_score"]
                metric_help = "silhouette_score: higher is better (cluster separation)."

            tune_metric = st.selectbox(
                "Optimization Metric",
                options=metric_options,
                help=metric_help,
                key="tune_metric"
            )

        with tune_col2:
            st.markdown("##### Data Configuration")

            available_symbols_tune = get_available_symbols()

            tune_symbols = st.multiselect(
                "Training Symbols",
                options=available_symbols_tune,
                default=available_symbols_tune[:min(3, len(available_symbols_tune))] if available_symbols_tune else [],
                key="tune_symbols"
            )

            from src.features.registry import feature_registry as _tune_fr
            _tune_all_features = _tune_fr.list_features()
            _tune_default = ["volatility", "log_return", "volume_ratio", "return_std", "price_range", "volatility_ratio"]
            _tune_default = [f for f in _tune_default if f in _tune_all_features]

            tune_features = st.multiselect(
                "Features",
                options=_tune_all_features,
                default=_tune_default,
                key="tune_features"
            )

            tune_date_col1, tune_date_col2 = st.columns(2)
            with tune_date_col1:
                tune_start_date = st.date_input(
                    "Start Date",
                    value=datetime.utcnow().date() - timedelta(days=180),
                    key="tune_start_date",
                )
            with tune_date_col2:
                tune_end_date = st.date_input(
                    "End Date",
                    value=datetime.utcnow().date(),
                    key="tune_end_date",
                )
            tune_test_split = st.slider("Test Split", min_value=0.1, max_value=0.4, value=0.2, key="tune_test_split")

        st.markdown("##### Parameter Grid")
        st.caption("Select which parameters to search over. All combinations will be evaluated.")

        param_grid: dict[str, list] = {}
        cluster_label = "n_clusters" if tune_model_type == "kmeans" else "n_components"

        grid_col1, grid_col2 = st.columns(2)

        with grid_col1:
            cluster_min, cluster_max = st.slider(
                f"Cluster/Component Range ({cluster_label})",
                min_value=2, max_value=20, value=(3, 8),
                key="tune_cluster_range"
            )
            param_grid[cluster_label] = list(range(cluster_min, cluster_max + 1))

            tune_contamination_values = st.multiselect(
                "Contamination Values",
                options=[0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20],
                default=[0.05],
                key="tune_contamination",
                help="Select one or more contamination rates to search."
            )
            if tune_contamination_values:
                param_grid['contamination'] = sorted(tune_contamination_values)

        with grid_col2:
            if tune_model_type == "gmm":
                tune_cov_types = st.multiselect(
                    "Covariance Types",
                    options=["full", "tied", "diag", "spherical"],
                    default=["full"],
                    key="tune_cov_types",
                    help="GMM covariance parameterizations to try."
                )
                if tune_cov_types:
                    param_grid['covariance_type'] = tune_cov_types

            tune_n_init_values = st.multiselect(
                "n_init Values",
                options=[1, 3, 5, 10, 15, 20],
                default=[10] if tune_model_type == "kmeans" else [5],
                key="tune_n_init",
                help="Number of random initializations to try."
            )
            if tune_n_init_values:
                param_grid['n_init'] = sorted(tune_n_init_values)

            if tune_model_type == "kmeans":
                tune_init_methods = st.multiselect(
                    "Initialization Method",
                    options=["k-means++", "random"],
                    default=[],
                    key="tune_init",
                    help="Centroid initialization strategies to try."
                )
                if tune_init_methods:
                    param_grid['init'] = tune_init_methods

                tune_algorithms = st.multiselect(
                    "Algorithm",
                    options=["lloyd", "elkan"],
                    default=[],
                    key="tune_algorithm",
                    help="K-Means algorithm variants to try."
                )
                if tune_algorithms:
                    param_grid['algorithm'] = tune_algorithms
            else:
                tune_init_params = st.multiselect(
                    "Initialization Method",
                    options=["kmeans", "k-means++", "random", "random_from_data"],
                    default=[],
                    key="tune_init_params",
                    help="GMM weight/mean initialization strategies to try."
                )
                if tune_init_params:
                    param_grid['init_params'] = tune_init_params

                tune_reg_covar = st.multiselect(
                    "Covariance Regularization (reg_covar)",
                    options=[1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3],
                    default=[],
                    key="tune_reg_covar",
                    format_func=lambda x: f"{x:.0e}",
                    help="Regularization values to try. Leave empty to use default (1e-6)."
                )
                if tune_reg_covar:
                    param_grid['reg_covar'] = sorted(tune_reg_covar)

        with st.expander("Additional Grid Parameters", expanded=False):
            adv_col1, adv_col2 = st.columns(2)

            with adv_col1:
                tune_max_iter_values = st.multiselect(
                    "max_iter Values",
                    options=[50, 100, 200, 300, 500, 1000],
                    default=[],
                    key="tune_max_iter",
                    help="Max iterations to try. Leave empty to use default."
                )
                if tune_max_iter_values:
                    param_grid['max_iter'] = sorted(tune_max_iter_values)

            with adv_col2:
                tune_scale_options = st.multiselect(
                    "scale_features",
                    options=[True, False],
                    default=[],
                    key="tune_scale",
                    format_func=lambda x: "Yes" if x else "No",
                    help="Whether to standardize features. Leave empty to use default (True)."
                )
                if tune_scale_options:
                    param_grid['scale_features'] = tune_scale_options

        # Feature subset search
        tune_search_features = st.checkbox(
            "Search over feature subsets",
            value=False,
            key="tune_search_features",
            help="Try different subsets of the selected features to find the best combination"
        )

        tune_feature_subsets = None
        n_tune_feature_combos = 1
        if tune_search_features and len(tune_features) >= 3:
            tune_min_subset = st.slider(
                "Minimum subset size",
                min_value=2,
                max_value=max(2, len(tune_features) - 1),
                value=max(2, len(tune_features) - 2),
                key="tune_min_subset"
            )

            from itertools import combinations as _combinations_tune
            tune_feature_subsets = []
            for size in range(tune_min_subset, len(tune_features) + 1):
                tune_feature_subsets.extend([list(c) for c in _combinations_tune(tune_features, size)])

            n_tune_feature_combos = len(tune_feature_subsets)
        elif tune_search_features:
            st.info("Select at least 3 features to enable feature subset search.")

        from sklearn.model_selection import ParameterGrid
        n_combinations = len(list(ParameterGrid(param_grid))) if param_grid else 0
        n_total_combinations = n_combinations * n_tune_feature_combos

        if n_tune_feature_combos > 1:
            st.info(f"**{n_combinations}** param combos x **{n_tune_feature_combos}** feature subsets = **{n_total_combinations}** total evaluations.")
        else:
            st.info(f"**{n_combinations}** parameter combinations will be evaluated.")

        with st.expander("Preview Parameter Grid", expanded=False):
            st.json({k: str(v) for k, v in param_grid.items()})

        can_tune = len(tune_symbols) > 0 and len(tune_features) > 0 and n_combinations > 0

        if n_total_combinations > 100:
            st.warning(f"Large search space ({n_total_combinations} total combinations). This may take a long time.")

        if st.button("Run Grid Search", type="primary", disabled=not can_tune, key="run_grid_search"):
            with st.spinner("Loading and preparing data..."):
                try:
                    from src.features.registry import compute_features, get_feature_matrix
                    from src.models.trainer import ModelTrainer

                    combined = load_training_data(
                        tune_symbols,
                        datetime.combine(tune_start_date, datetime.min.time()),
                        datetime.combine(tune_end_date, datetime.max.time()),
                    )
                    if combined.empty:
                        st.error("No data loaded. Check that selected symbols have data.")
                        st.stop()

                    df_features = compute_features(combined, tune_features)
                    X, df_clean = get_feature_matrix(df_features, tune_features)

                    split_idx = int(len(X) * (1 - tune_test_split))
                    X_train, X_test = X[:split_idx], X[split_idx:]

                    st.info(f"Data prepared: {len(X_train):,} train / {len(X_test):,} test samples, {len(tune_features)} features")
                except Exception as e:
                    st.error(f"Data preparation failed: {e}")
                    st.stop()

            tune_progress_bar = st.progress(0, text=f"0/{n_total_combinations} combinations completed")

            try:
                from config.settings import settings
                trainer = ModelTrainer(
                    mlflow_tracking_uri=settings.mlflow_tracking_uri,
                    use_mlflow=True
                )

                def _tune_progress(current, total, result):
                    pct = current / total
                    score_text = ""
                    if result and tune_metric in result:
                        score_text = f" — latest {tune_metric}: {result[tune_metric]:.4f}"
                    tune_progress_bar.progress(pct, text=f"{current}/{total} combinations completed{score_text}")

                tuning_results = trainer.train_with_tuning(
                    model_type=tune_model_type,
                    X_train=X_train,
                    feature_names=tune_features,
                    param_grid=param_grid,
                    feature_subsets=tune_feature_subsets,
                    X_test=X_test,
                    metric=tune_metric,
                    tags={
                        'symbols': ','.join(tune_symbols),
                        'source': 'streamlit_grid_search'
                    },
                    on_progress=_tune_progress,
                )

                best_model = tuning_results['best_model']
                best_score = tuning_results['best_score']
                best_run_id = tuning_results['best_run_id']
                all_results = tuning_results['all_results']

                if best_model is None:
                    st.warning("Grid search completed but no valid results were found.")
                    st.stop()

                st.success(f"Grid search complete! Best {tune_metric}: **{best_score:.4f}**")

                if best_run_id:
                    st.info(f"Best MLflow Run ID: `{best_run_id}`")

                st.markdown("##### Best Parameters")
                best_result = next((r for r in all_results if r['run_id'] == best_run_id), None)
                if best_result:
                    st.json(best_result['params'])

                # Show best features if feature subset search was used
                best_features = tuning_results.get('best_features')
                if best_features and tune_feature_subsets:
                    st.markdown(f"**Best Features** ({len(best_features)}): `{', '.join(best_features)}`")

                st.markdown("##### All Results")
                results_rows = []
                for r in all_results:
                    row = {**r['params']}
                    row[tune_metric] = r.get(tune_metric)
                    row['threshold'] = r.get('threshold')
                    if 'features' in r:
                        row['n_features'] = r.get('n_features')
                        row['features'] = ', '.join(r.get('features', []))
                    row['run_id'] = r.get('run_id', '')[:8] if r.get('run_id') else ''
                    results_rows.append(row)

                results_df = pd.DataFrame(results_rows)
                if tune_metric in results_df.columns:
                    results_df = results_df.sort_values(tune_metric, ascending=(tune_metric == 'bic'))
                st.dataframe(results_df, use_container_width=True)

                st.markdown("##### Results Visualization")
                if tune_metric in results_df.columns and cluster_label in results_df.columns:
                    viz_col1, viz_col2 = st.columns(2)

                    with viz_col1:
                        fig = px.scatter(
                            results_df,
                            x=cluster_label,
                            y=tune_metric,
                            color='contamination' if 'contamination' in results_df.columns else None,
                            title=f"{tune_metric} vs {cluster_label}",
                            hover_data=results_df.columns.tolist()
                        )
                        if best_result:
                            fig.add_trace(go.Scatter(
                                x=[best_result['params'].get(cluster_label)],
                                y=[best_result.get(tune_metric)],
                                mode='markers',
                                marker=dict(size=15, symbol='star', color='red', line=dict(width=2, color='black')),
                                name='Best',
                                showlegend=True
                            ))
                        st.plotly_chart(fig, use_container_width=True)

                    with viz_col2:
                        fig = px.box(
                            results_df,
                            x=cluster_label,
                            y=tune_metric,
                            title=f"{tune_metric} Distribution by {cluster_label}"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                st.markdown("##### Best Model Performance")
                model_params = best_model.get_model_params()
                perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)

                # Slice to best feature subset if needed
                best_features = tuning_results.get('best_features')
                if best_features and best_features != tune_features:
                    best_col_idx = [tune_features.index(f) for f in best_features]
                    X_train_best = X_train[:, best_col_idx]
                    X_test_best = X_test[:, best_col_idx]
                else:
                    X_train_best = X_train
                    X_test_best = X_test

                with perf_col1:
                    sil = model_params.get('silhouette_score')
                    st.metric("Silhouette Score", f"{sil:.4f}" if sil else "N/A")
                with perf_col2:
                    st.metric("Threshold", f"{best_model.threshold:.4f}")
                with perf_col3:
                    train_preds = best_model.predict(X_train_best)
                    st.metric("Train Anomaly Rate", f"{train_preds.mean():.2%}")
                with perf_col4:
                    test_preds = best_model.predict(X_test_best)
                    st.metric("Test Anomaly Rate", f"{test_preds.mean():.2%}")

            except Exception as e:
                st.error(f"Grid search failed: {e}")
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
