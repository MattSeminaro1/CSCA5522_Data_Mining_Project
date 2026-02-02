"""
Crypto Anomaly Detection - Streamlit Dashboard.

Main entry point for the application UI.
"""

import streamlit as st

st.set_page_config(
    page_title="Crypto Anomaly Detection",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded"
)


def check_database_status() -> tuple[bool, str]:
    """Check if database is connected and has data."""
    try:
        from src.data.database import db
        if not db.test_connection():
            return False, "Connection failed"
        
        status = db.get_data_status()
        if status.empty:
            return True, "Connected (no data)"
        return True, f"Connected ({len(status)} symbols)"
    except Exception as e:
        return False, str(e)[:50]


def check_mlflow_status() -> tuple[bool, str]:
    """Check if MLflow is accessible."""
    try:
        import mlflow
        from config.settings import settings
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        return True, "Connected"
    except Exception as e:
        return False, str(e)[:50]


def main():
    """Main application entry point."""
    
    # Sidebar
    with st.sidebar:
        st.title("Crypto Anomaly Detection")
        st.markdown("---")
        
        st.markdown("""
        ### Navigation
        Use the pages in the sidebar to:
        
        1. **Data Management** - Download and manage historical data
        2. **Training** - Train and tune models
        3. **Evaluation** - Compare model performance
        4. **Live Detection** - Real-time anomaly monitoring
        """)
        
        st.markdown("---")
        
        # System status
        st.subheader("System Status")
        
        db_ok, db_msg = check_database_status()
        if db_ok:
            st.success(f"Database: {db_msg}")
        else:
            st.error(f"Database: {db_msg}")
        
        mlflow_ok, mlflow_msg = check_mlflow_status()
        if mlflow_ok:
            st.success(f"MLflow: {mlflow_msg}")
        else:
            st.warning(f"MLflow: {mlflow_msg}")
        
        st.markdown("---")
        st.caption("v1.0.0")
    
    # Main content
    st.title("Welcome to Crypto Anomaly Detection")
    
    st.markdown("""
    This application provides real-time anomaly detection for cryptocurrency markets 
    using machine learning clustering algorithms.
    
    ## Getting Started
    
    ### 1. Load Data
    Navigate to **Data Management** to download historical cryptocurrency data.
    The system will fetch data from Binance's public data repository.
    
    ### 2. Train Models
    Go to **Training** to train anomaly detection models:
    - **K-Means**: Fast, interpretable clustering
    - **GMM**: Probabilistic soft clustering
    
    ### 3. Evaluate
    Use **Evaluation** to compare model performance and select the best one.
    
    ### 4. Monitor
    Finally, **Live Detection** shows real-time anomaly detection on streaming data.
    
    ---
    """)
    
    # Dashboard overview
    st.subheader("Quick Stats")
    
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        from src.data.database import db
        status = db.get_data_status()
        
        with col1:
            st.metric("Symbols", len(status) if not status.empty else 0)
        
        with col2:
            total_rows = int(status['total_rows'].sum()) if not status.empty else 0
            st.metric("Total Candles", f"{total_rows:,}")
        
        with col3:
            try:
                import mlflow
                client = mlflow.tracking.MlflowClient()
                models = list(client.search_registered_models())
                st.metric("Registered Models", len(models))
            except Exception:
                st.metric("Registered Models", "N/A")
        
        with col4:
            try:
                anomalies = db.get_recent_anomalies(hours=24, limit=100)
                st.metric("Anomalies (24h)", len(anomalies))
            except Exception:
                st.metric("Anomalies (24h)", "N/A")
    
    except Exception as e:
        for col in [col1, col2, col3, col4]:
            with col:
                st.metric("N/A", "N/A")
        st.warning(f"Could not load stats: {e}")
    
    # Data status table
    st.markdown("---")
    st.subheader("Data Coverage")
    
    try:
        from src.data.database import db
        status = db.get_data_status()
        
        if not status.empty:
            st.dataframe(status, use_container_width=True, hide_index=True)
        else:
            st.info("No data loaded yet. Go to Data Management to download historical data.")
    except Exception:
        st.info("Start the infrastructure with `docker-compose up` to see data status.")


if __name__ == "__main__":
    main()
