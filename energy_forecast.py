"""
Airflow DAG - Energy Demand Forecasting Pipeline
Author: Corey Leath
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import src.data.preprocess as preprocess
import src.features.build_features as features
import src.models.train as train
import src.models.evaluate as evaluate

# Default arguments for DAG tasks
default_args = {
    'owner': 'corey',
    'depends_on_past': False,
    'email_on_failure': True,
    'retries': 1,
}

# Define the DAG
with DAG(
    dag_id='energy_demand_forecast',
    default_args=default_args,
    description='Energy Demand Forecasting Pipeline',
    schedule_interval='@daily',
    start_date=days_ago(1),
    catchup=False,
    tags=['devmlops', 'energy', 'forecasting'],
) as dag:

    # Task 1 - Data Preprocessing
    preprocess_data = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess.main,
    )

    # Task 2 - Feature Engineering
    build_features = PythonOperator(
        task_id='build_features',
        python_callable=features.main,
    )

    # Task 3 - Train Model
    train_model = PythonOperator(
        task_id='train_model',
        python_callable=lambda: train.main(config_path='configs/train.yaml'),
    )

    # Task 4 - Evaluate Model
    evaluate_model = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate.main,
    )

    # Define task dependencies
    preprocess_data >> build_features >> train_model >> evaluate_model
