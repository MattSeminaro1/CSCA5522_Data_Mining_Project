# Crypto Anomaly Detection

Real-time anomaly detection for cryptocurrency markets using unsupervised machine learning.

## Overview

This system detects anomalous price movements in cryptocurrency markets using clustering algorithms (K-Means and Gaussian Mixture Models). It features:

- Historical data collection from Binance public data repository
- Feature engineering with technical indicators
- Model training with hyperparameter tuning
- MLflow experiment tracking and model registry
- Real-time detection via streaming inference
- Streamlit dashboard for monitoring and analysis

## Quick Start

### Prerequisites

- Docker Desktop
- 8GB RAM minimum (16GB recommended)

### 1. Start Infrastructure

```bash
cd crypto_anomaly/infrastructure
docker-compose up -d
```

Wait for services to be healthy:

```bash
docker-compose ps
```

### 2. Access the Dashboard

Open http://localhost:8501 in your browser.

### 3. Download Data

1. Navigate to "Data Management"
2. Select symbols (default: BTC, ETH, SOL, AVAX, LINK)
3. Choose number of days (365 recommended for initial training)
4. Click "Start Download"

### 4. Train a Model

1. Navigate to "Training"
2. Select model type (K-Means or GMM)
3. Choose number of clusters (5-10 recommended)
4. Select features and training data
5. Click "Train Model"

### 5. Evaluate and Monitor

- Use "Evaluation" to compare model performance
- Use "Live Detection" to run anomaly detection on historical replay

## Services

| Service | URL | Credentials |
|---------|-----|-------------|
| Streamlit | http://localhost:8501 | None |
| MLflow | http://localhost:5001 | None |
| MinIO | http://localhost:9001 | minioadmin / minioadmin123 |
| TimescaleDB | localhost:5432 | crypto / crypto123 |
| Redis | localhost:6379 | None |

## Project Structure

```
crypto_anomaly/
├── infrastructure/          # Docker and database setup
├── src/
│   ├── data/               # Data collection and storage
│   ├── features/           # Feature engineering
│   ├── models/             # ML models and training
│   ├── streaming/          # Real-time inference
│   └── evaluation/         # Metrics and analysis
├── flows/                  # Prefect workflows
├── app/                    # Streamlit dashboard
├── config/                 # Configuration
└── tests/                  # Unit tests
```

## Features Computed

| Feature | Description |
|---------|-------------|
| volatility | Intra-candle price range normalized by open |
| log_return | Log return of close price |
| volume_ratio | Volume relative to 10-period moving average |
| volatility_ratio | Volatility relative to 10-period MA |
| return_std | Rolling standard deviation of returns |
| price_range | High-low spread normalized by close |

## Model Types

### K-Means Clustering

- Anomaly score: Distance to nearest cluster centroid
- Fast training and inference
- Hard cluster assignments

### Gaussian Mixture Model (GMM)

- Anomaly score: Negative log-likelihood
- Soft cluster assignments (probabilities)
- Handles elliptical cluster shapes

## Configuration

Set environment variables or create `.env` file:

```bash
DATABASE_URL=postgresql://crypto:crypto123@localhost:5432/crypto_anomaly
REDIS_URL=redis://localhost:6379
MLFLOW_TRACKING_URI=http://localhost:5001
```

## CLI Usage

```bash
# Enter the container
docker exec -it crypto_app bash

# Download data
python -m flows.data_collection collect 365

# Update data (incremental)
python -m flows.data_collection update

# Compute features
python -m flows.data_collection features 30
```

## Stopping Services

```bash
cd infrastructure
docker-compose down

# To remove all data:
docker-compose down -v
```

## Troubleshooting

### Services not starting

Check logs:
```bash
docker-compose logs timescaledb
docker-compose logs mlflow
```

### Database connection issues

Ensure TimescaleDB is healthy:
```bash
docker-compose ps timescaledb
```

### Slow data download

- Network issues with data.binance.vision
- Reduce number of symbols or days
- Check disk space

## License

MIT
