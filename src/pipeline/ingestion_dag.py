from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'portfolio-manager',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'market_data_ingestion',
    default_args=default_args,
    description='Fetches market data from yfinance and versions with DVC',
    schedule_interval=timedelta(days=1),
    catchup=False,
) as dag:

    fetch_data = BashOperator(
        task_id='fetch_market_data',
        bash_command='python3 /opt/airflow/dags/ingest.py',
    )

    # Note: DVC tracking task will be added here once we've finalized
    # how to pass the host's DVC credentials/config into the container.
    
    fetch_data
