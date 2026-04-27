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
    'retry_delay': timedelta(seconds=10),
}

with DAG(
    'market_data_ingestion',
    default_args=default_args,
    description='Fetches, validates, and engineers features with granular DVC tracking',
    schedule_interval=timedelta(days=1),
    catchup=False,
) as dag:

    # 1. Ingestion
    fetch_data = BashOperator(
        task_id='fetch_market_data',
        bash_command='python3 /opt/airflow/dags/ingest.py',
        cwd='/opt/airflow'
    )

    track_raw = BashOperator(
        task_id='track_raw_data',
        bash_command='dvc add data/raw',
        cwd='/opt/airflow'
    )

    # 2. Validation
    validate_data = BashOperator(
        task_id='validate_market_data',
        bash_command='python3 /opt/airflow/dags/validate.py',
        cwd='/opt/airflow'
    )

    track_validated = BashOperator(
        task_id='track_validated_data',
        bash_command='dvc add data/raw',
        cwd='/opt/airflow'
    )

    # 3. Feature Engineering
    engineer_features = BashOperator(
        task_id='feature_engineering',
        bash_command='python3 /opt/airflow/dags/features.py',
        cwd='/opt/airflow'
    )

    track_features = BashOperator(
        task_id='track_features_data',
        bash_command='dvc add data/features',
        cwd='/opt/airflow'
    )

    # 4. Model Training
    model_training = BashOperator(
        task_id='model_training',
        bash_command='python3 /opt/airflow/dags/train.py',
        cwd='/opt/airflow'
    )
    
    # Dependency Chain with intermediate tracking
    fetch_data >> track_raw >> validate_data >> track_validated >> engineer_features >> track_features >> model_training
