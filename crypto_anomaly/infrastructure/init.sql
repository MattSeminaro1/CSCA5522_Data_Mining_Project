-- TimescaleDB Schema for Crypto Anomaly Detection
-- This script runs automatically when the container starts for the first time.

CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Raw OHLCV candlestick data from exchanges
CREATE TABLE IF NOT EXISTS raw_ohlcv (
    time            TIMESTAMPTZ NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    open            DOUBLE PRECISION NOT NULL,
    high            DOUBLE PRECISION NOT NULL,
    low             DOUBLE PRECISION NOT NULL,
    close           DOUBLE PRECISION NOT NULL,
    volume          DOUBLE PRECISION NOT NULL,
    quote_volume    DOUBLE PRECISION,
    trade_count     INTEGER,
    source          VARCHAR(50) DEFAULT 'binance',
    ingested_at     TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (time, symbol)
);

SELECT create_hypertable('raw_ohlcv', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_raw_ohlcv_symbol_time 
ON raw_ohlcv (symbol, time DESC);

-- Enable compression (wrapped in DO block for error handling)
DO $$
BEGIN
    ALTER TABLE raw_ohlcv SET (
        timescaledb.compress,
        timescaledb.compress_segmentby = 'symbol'
    );
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Compression already enabled or not supported';
END $$;

-- Add compression policy
DO $$
BEGIN
    PERFORM add_compression_policy('raw_ohlcv', INTERVAL '7 days', if_not_exists => TRUE);
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Compression policy already exists or failed';
END $$;


-- Computed features for model training and inference
CREATE TABLE IF NOT EXISTS features (
    time                TIMESTAMPTZ NOT NULL,
    symbol              VARCHAR(20) NOT NULL,
    volatility          DOUBLE PRECISION,
    log_return          DOUBLE PRECISION,
    price_range         DOUBLE PRECISION,
    volume_ratio        DOUBLE PRECISION,
    volatility_ratio    DOUBLE PRECISION,
    return_std          DOUBLE PRECISION,
    volume_ma           DOUBLE PRECISION,
    volatility_ma       DOUBLE PRECISION,
    feature_version     VARCHAR(20) DEFAULT 'v1',
    computed_at         TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (time, symbol)
);

SELECT create_hypertable('features', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_features_symbol_time 
ON features (symbol, time DESC);


-- Model predictions for monitoring and analysis
CREATE TABLE IF NOT EXISTS predictions (
    id                  BIGSERIAL,
    time                TIMESTAMPTZ NOT NULL,
    symbol              VARCHAR(20) NOT NULL,
    model_name          VARCHAR(100) NOT NULL,
    model_version       VARCHAR(50) NOT NULL,
    model_run_id        VARCHAR(50),
    anomaly_score       DOUBLE PRECISION NOT NULL,
    is_anomaly          BOOLEAN NOT NULL,
    threshold_used      DOUBLE PRECISION NOT NULL,
    cluster_id          INTEGER,
    close_price         DOUBLE PRECISION,
    features_json       JSONB,
    latency_ms          INTEGER,
    source              VARCHAR(20) DEFAULT 'streaming',
    predicted_at        TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (id, time)
);

SELECT create_hypertable('predictions', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_predictions_anomalies 
ON predictions (symbol, time DESC) 
WHERE is_anomaly = TRUE;

CREATE INDEX IF NOT EXISTS idx_predictions_model 
ON predictions (model_name, model_version, time DESC);


-- Tracking for data collection jobs
CREATE TABLE IF NOT EXISTS collection_jobs (
    id                  SERIAL PRIMARY KEY,
    symbol              VARCHAR(20) NOT NULL,
    start_date          DATE NOT NULL,
    end_date            DATE NOT NULL,
    interval_type       VARCHAR(10) DEFAULT '1m',
    status              VARCHAR(20) DEFAULT 'pending',
    rows_collected      INTEGER DEFAULT 0,
    error_message       TEXT,
    started_at          TIMESTAMPTZ,
    completed_at        TIMESTAMPTZ,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_collection_jobs_symbol 
ON collection_jobs (symbol, status);


-- Trained model metadata (supplements MLflow)
CREATE TABLE IF NOT EXISTS model_metadata (
    id                  SERIAL PRIMARY KEY,
    model_name          VARCHAR(100) NOT NULL,
    model_version       VARCHAR(50) NOT NULL,
    mlflow_run_id       VARCHAR(50),
    model_type          VARCHAR(50) NOT NULL,
    hyperparameters     JSONB,
    feature_names       JSONB,
    training_symbols    JSONB,
    training_start      TIMESTAMPTZ,
    training_end        TIMESTAMPTZ,
    training_samples    INTEGER,
    silhouette_score    DOUBLE PRECISION,
    inertia             DOUBLE PRECISION,
    bic                 DOUBLE PRECISION,
    contamination       DOUBLE PRECISION,
    threshold           DOUBLE PRECISION,
    is_active           BOOLEAN DEFAULT FALSE,
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (model_name, model_version)
);


-- View: Data status per symbol
CREATE OR REPLACE VIEW v_data_status AS
SELECT 
    symbol,
    MIN(time) AS earliest,
    MAX(time) AS latest,
    COUNT(*) AS total_rows,
    MAX(time) - MIN(time) AS time_span
FROM raw_ohlcv
GROUP BY symbol
ORDER BY symbol;


-- View: Daily anomaly summary
CREATE OR REPLACE VIEW v_daily_anomaly_summary AS
SELECT 
    time_bucket('1 day', time) AS day,
    symbol,
    model_name,
    COUNT(*) FILTER (WHERE is_anomaly) AS anomaly_count,
    COUNT(*) AS total_predictions,
    ROUND(AVG(anomaly_score)::NUMERIC, 4) AS avg_score,
    MAX(anomaly_score) AS max_score
FROM predictions
GROUP BY 1, 2, 3
ORDER BY 1 DESC, 2, 3;


-- View: Recent anomalies in last 24 hours
CREATE OR REPLACE VIEW v_recent_anomalies AS
SELECT 
    p.time,
    p.symbol,
    p.model_name,
    p.anomaly_score,
    p.threshold_used,
    p.close_price,
    p.latency_ms
FROM predictions p
WHERE p.is_anomaly = TRUE
  AND p.time > NOW() - INTERVAL '24 hours'
ORDER BY p.time DESC
LIMIT 1000;


-- Function: Get latest candles for a symbol
CREATE OR REPLACE FUNCTION get_latest_candles(
    p_symbol VARCHAR(20),
    p_limit INTEGER DEFAULT 100
)
RETURNS TABLE (
    candle_time TIMESTAMPTZ,
    candle_open DOUBLE PRECISION,
    candle_high DOUBLE PRECISION,
    candle_low DOUBLE PRECISION,
    candle_close DOUBLE PRECISION,
    candle_volume DOUBLE PRECISION
) AS $$
BEGIN
    RETURN QUERY
    SELECT r.time, r.open, r.high, r.low, r.close, r.volume
    FROM raw_ohlcv r
    WHERE r.symbol = p_symbol
    ORDER BY r.time DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;


-- Function: Get data range for a symbol
CREATE OR REPLACE FUNCTION get_data_range(p_symbol VARCHAR(20))
RETURNS TABLE (
    min_time TIMESTAMPTZ,
    max_time TIMESTAMPTZ,
    row_count BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT MIN(r.time), MAX(r.time), COUNT(*)
    FROM raw_ohlcv r
    WHERE r.symbol = p_symbol;
END;
$$ LANGUAGE plpgsql;


-- Continuous aggregate: Hourly OHLCV rollup
DO $$
BEGIN
    CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_hourly
    WITH (timescaledb.continuous) AS
    SELECT 
        time_bucket('1 hour', time) AS bucket,
        symbol,
        FIRST(open, time) AS open,
        MAX(high) AS high,
        MIN(low) AS low,
        LAST(close, time) AS close,
        SUM(volume) AS volume,
        SUM(trade_count) AS trade_count
    FROM raw_ohlcv
    GROUP BY bucket, symbol
    WITH NO DATA;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'ohlcv_hourly already exists';
END $$;

DO $$
BEGIN
    PERFORM add_continuous_aggregate_policy('ohlcv_hourly',
        start_offset => INTERVAL '3 hours',
        end_offset => INTERVAL '1 hour',
        schedule_interval => INTERVAL '1 hour');
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'ohlcv_hourly policy already exists or failed';
END $$;


-- Continuous aggregate: Daily OHLCV rollup
DO $$
BEGIN
    CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_daily
    WITH (timescaledb.continuous) AS
    SELECT 
        time_bucket('1 day', time) AS bucket,
        symbol,
        FIRST(open, time) AS open,
        MAX(high) AS high,
        MIN(low) AS low,
        LAST(close, time) AS close,
        SUM(volume) AS volume,
        SUM(trade_count) AS trade_count
    FROM raw_ohlcv
    GROUP BY bucket, symbol
    WITH NO DATA;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'ohlcv_daily already exists';
END $$;

DO $$
BEGIN
    PERFORM add_continuous_aggregate_policy('ohlcv_daily',
        start_offset => INTERVAL '3 days',
        end_offset => INTERVAL '1 day',
        schedule_interval => INTERVAL '1 day');
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'ohlcv_daily policy already exists or failed';
END $$;
