from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pickle
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-experiment")

models_folder = Path('/opt/airflow/models')
models_folder.mkdir(exist_ok=True)

def read_dataframe(year, month):
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)
    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    df['PU_DO'] = df['PULocationID'].astype(str) + '_' + df['DOLocationID'].astype(str)
    return df

def prepare_datasets(year, month, **context):
    df_train = read_dataframe(year, month)
    df_val = read_dataframe(year if month < 12 else year + 1, month + 1 if month < 12 else 1)

    dv = DictVectorizer()
    X_train = dv.fit_transform(df_train[['PU_DO', 'trip_distance']].to_dict(orient='records'))
    X_val = dv.transform(df_val[['PU_DO', 'trip_distance']].to_dict(orient='records'))

    y_train = df_train['duration'].values
    y_val = df_val['duration'].values

    # Save to XComs
    context['ti'].xcom_push(key='X_train', value=X_train)
    context['ti'].xcom_push(key='X_val', value=X_val)
    context['ti'].xcom_push(key='y_train', value=y_train)
    context['ti'].xcom_push(key='y_val', value=y_val)
    context['ti'].xcom_push(key='dv', value=dv)

def train_model_task(**context):
    X_train = context['ti'].xcom_pull(key='X_train')
    X_val = context['ti'].xcom_pull(key='X_val')
    y_train = context['ti'].xcom_pull(key='y_train')
    y_val = context['ti'].xcom_pull(key='y_val')
    dv = context['ti'].xcom_pull(key='dv')

    with mlflow.start_run() as run:
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        params = {
            'learning_rate': 0.1,
            'max_depth': 6,
            'objective': 'reg:squarederror',
            'seed': 42,
        }

        booster = xgb.train(params, train, evals=[(valid, 'val')])
        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)

        mlflow.log_params(params)
        mlflow.log_metric("rmse", rmse)

        with open("/opt/airflow/models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("/opt/airflow/models/preprocessor.b", artifact_path="preprocessor")

        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

default_args = {
    'start_date': datetime(2024, 1, 1),
}

with DAG(
    'train_taxi_model_dag',
    schedule_interval=None,
    catchup=False,
    default_args=default_args,
) as dag:

    prepare = PythonOperator(
        task_id='prepare_datasets',
        python_callable=prepare_datasets,
        op_kwargs={'year': 2023, 'month': 1},
    )

    train = PythonOperator(
        task_id='train_model',
        python_callable=train_model_task,
    )

    prepare >> train
