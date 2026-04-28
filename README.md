# 🚀 Portfolio AI: End-to-End MLOps Pipeline

An automated system for stock market data ingestion, validation, feature engineering, ML experimentation, and portfolio optimization.

## 🏗️ System Architecture

- **Data Pipeline:** Apache Airflow (DAGs for automated ETL)
- **Data Versioning:** DVC (Data Version Control)
- **Experiment Tracking:** MLflow (Model registry & hyperparameter logging)
- **Backend:** FastAPI (Model serving & Portfolio Optimization)
- **Frontend:** Streamlit (Interactive Plotly dashboard)
- **Reverse Proxy:** Nginx
- **Database:** PostgreSQL (Airflow metadata)

---

## 🛠️ Quick Start

### 1. Prerequisites
- Docker & Docker Compose
- Git
- (Optional) DVC for local tracking

### 2. Build and Run
Execute the following commands in your terminal:

```bash
# Enable BuildKit for faster cached builds
export DOCKER_BUILDKIT=1

# Build and start all services in the background
docker compose up -d --build
```

### 3. Initialize Data (First Run)
Once the containers are up, trigger the Airflow DAG to fetch historical data and train the initial model:
1. Open [http://localhost:8081](http://localhost:8081)
2. Login with `admin` / `admin`
3. Unpause and trigger the `market_data_ingestion` DAG.

---

## 🌐 Service Map

| Service | URL | Description |
| :--- | :--- | :--- |
| **Streamlit Dashboard** | [http://localhost](http://localhost) | Main UI for visualization & predictions |
| **Airflow UI** | [http://localhost:8081](http://localhost:8081) | DAG management & Pipeline monitoring |
| **MLflow UI** | [http://localhost:5000](http://localhost:5000) | Experiment tracking & Model Registry |
| **FastAPI Backend** | [http://localhost/api](http://localhost/api) | Model inference & Optimization engine |

---

## 🔍 API Endpoints

### `POST /predict`
Submit technical indicators to get a price movement prediction.
```json
{
  "ticker": "AAPL",
  "RSI": 55.5,
  "MACD": 1.2,
  "MACD_Signal": 0.8,
  "BB_Upper": 150.0,
  "BB_Lower": 140.0,
  "SMA_20": 145.0,
  "SMA_50": 142.0
}
```

### `POST /portfolio`
Get optimal weight allocation based on Mean-Variance Optimization (MVO).
```json
{
  "risk_aversion": 1.5
}
```

### `POST /history`
Fetch processed historical data and technical indicators for charts.

---

## 📁 Project Structure
- `src/pipeline/`: Airflow DAGs and ETL scripts.
- `src/api/`: FastAPI backend logic and MVO optimization.
- `src/app/`: Streamlit frontend code.
- `data/`: Local storage for raw and processed data (DVC tracked).
- `mlruns/`: MLflow artifact storage and SQLite database.
