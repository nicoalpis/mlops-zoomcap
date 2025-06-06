import os
import pickle
import mlflow
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
import numpy as np

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def train_and_log_model(df, *args, **kwargs):
    """
    Train a Linear Regression model and log it with MLflow.
    """
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    y_train = df['duration'].values

    with mlflow.start_run():
        lr = LinearRegression()
        lr.fit(X_train, y_train)

        y_pred = lr.predict(X_train)
        rmse = np.sqrt(mean_squared_error(y_train, y_pred))
        mlflow.log_metric('rmse', rmse)

        # Save model
        model_path = 'homework03_models/lin_reg.bin'
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f_out:
            pickle.dump((dv, lr), f_out)

        mlflow.log_artifact(model_path, artifact_path='homework03_models')

        print(f"Model intercept: {lr.intercept_:.2f}")
        print(f"RMSE: {rmse:.2f}")

    return lr, dv


@test
def test_output(output, *args) -> None:
    model, vectorizer = output
    assert hasattr(model, 'predict'), 'Model is not trained properly'
    assert hasattr(vectorizer, 'transform'), 'Vectorizer not fitted'
