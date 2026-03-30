import pickle
from src.logging_config import logger
import os
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn.ensemble import GradientBoostingClassifier
import mlflow
import json
import dagshub

# Initialize DagsHub + MLflow
dagshub.init(repo_owner='umiii-786', repo_name='placement-prediction-Model', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/umiii-786/placement-prediction-Model.mlflow")


def load_data(data_path: str) -> tuple:
    try:
        logger.info(f"Loading data from path: {data_path}")

        train_path = os.path.join(data_path, 'train_ds.csv')
        test_path = os.path.join(data_path, 'test_ds.csv')

        train_ds = pd.read_csv(train_path)
        test_ds = pd.read_csv(test_path)

        logger.info("Data loaded successfully")
        logger.debug(f"Train shape: {train_ds.shape}, Test shape: {test_ds.shape}")

        return train_ds, test_ds

    except Exception:
        logger.exception("Error occurred while loading data")
        raise


def load_model(model_path: str):
    try:
        logger.info(f"Loading model from path: {model_path}")

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        logger.info("Model loaded successfully")
        return model

    except Exception:
        logger.exception("Error occurred while loading model")
        raise


def get_ids(ids_path: str):
    try:
        logger.info(f"Loading run/model IDs from: {ids_path}")

        with open(ids_path, 'r') as file:
            data = json.load(file)

        logger.info(f"IDs loaded successfully: {data}")
        return data

    except Exception:
        logger.exception("Error occurred while loading IDs JSON")
        raise


def test_model(model: GradientBoostingClassifier,
               train_ds: pd.DataFrame,
               test_ds: pd.DataFrame):
    try:
        logger.info("Starting model evaluation")

        # Split features & target
        x_train = train_ds.drop('PlacementStatus', axis=1)
        x_test = test_ds.drop('PlacementStatus', axis=1)

        y_train = train_ds['PlacementStatus']
        y_test = test_ds['PlacementStatus']

        logger.info("Feature-target split completed")

        # Predictions
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)

        logger.info("Predictions completed")

        # Load run_id
        ids_path = os.path.join('reports', 'data.json')
        ids = get_ids(ids_path)

        logger.info(f"Using MLflow run_id: {ids['run_id']}")

        # MLflow logging
        with mlflow.start_run(run_id=ids['run_id']):
            logger.info("Logging metrics to MLflow")

            # Train metrics
            mlflow.log_metric("train_accuracy", accuracy_score(y_train, y_train_pred))
            mlflow.log_metric("train_recall", recall_score(y_train, y_train_pred))
            mlflow.log_metric("train_precision", precision_score(y_train, y_train_pred))
            mlflow.log_metric("train_f1", f1_score(y_train, y_train_pred))

            # Test metrics
            mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_test_pred))
            mlflow.log_metric("test_recall", recall_score(y_test, y_test_pred))
            mlflow.log_metric("test_precision", precision_score(y_test, y_test_pred))
            mlflow.log_metric("test_f1", f1_score(y_test, y_test_pred))

            logger.info("Metrics logged successfully")

    except Exception:
        logger.exception("Error occurred during model evaluation")
        raise


def main() -> None:
    try:
        logger.info("Evaluation pipeline started")

        model_path = os.path.join('models', 'gradient_boosting_model.pkl')
        data_path = os.path.join('data', 'processed')

        # Load model
        model = load_model(model_path)

        # Load data
        train_ds, test_ds = load_data(data_path)

        # Evaluate model
        test_model(model, train_ds, test_ds)

        logger.info("Evaluation pipeline completed successfully")

    except Exception:
        logger.exception("Pipeline failed")
        raise


if __name__ == "__main__":
    main()