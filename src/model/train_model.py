import pandas as pd
import os
from src.logging_config import logger
from sklearn.ensemble import GradientBoostingClassifier
import mlflow
from mlflow.models import infer_signature
import json
import dagshub
import pickle

# Initialize DagsHub + MLflow

# dagshub_pat=os.getenv("DAGSHUB_PAT")
dagshub_pat="a55ae4d7356bf84fa662753c4cff9084c43da67d"
if not dagshub_pat:
    raise EnvironmentError('DAGSHUB_PAT environment variable is not setted ') 
os.environ['MLFLOW_TRACKING_USERNAME']=dagshub_pat 
os.environ['MLFLOW_TRACKING_PASSWORD']=dagshub_pat 

mlflow.set_tracking_uri("https://dagshub.com/umiii-786/placement-prediction-Model.mlflow")

def load_data(data_path: str) -> pd.DataFrame:
    try:
        logger.info(f"Loading data from path: {data_path}")

        train_path = os.path.join(data_path, 'train_ds.csv')
        train_ds = pd.read_csv(train_path)

        logger.info("Data loaded successfully")
        logger.debug(f"Train shape: {train_ds.shape}")

        return train_ds

    except Exception:
        logger.exception("Error occurred while loading data")
        raise


def train_model(x_train, y_train, params) -> GradientBoostingClassifier:
    try:
        logger.info("Starting model training")

        model = GradientBoostingClassifier(**params)
        model.fit(x_train, y_train)

        logger.info("Model training completed successfully")

        return model

    except Exception:
        logger.exception("Error occurred during model training")
        raise


def save_model_pickle(model, model_name: str):
    try:
        logger.info("Saving model as pickle file")

        # Create models directory if not exists
        os.makedirs('models', exist_ok=True)

        file_path = os.path.join('models', f"{model_name}.pkl")

        with open(file_path, 'wb') as f:
            pickle.dump(model, f)

        logger.info(f"Model saved successfully at {file_path}")

    except Exception:
        logger.exception("Error occurred while saving model as pickle")
        raise

def log_model_and_parameters(model, parameters, signature):
    try:
        logger.info("Starting MLflow logging")

        mlflow.set_experiment(experiment_name='Pipeline Result')

        with mlflow.start_run() as run:
            logger.info(f"MLflow run started with run_id: {run.info.run_id}")

            # Log parameters
            mlflow.log_params(parameters)
            logger.info("Parameters logged successfully")

            # Log model
            logged_model = mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model", 
                    signature=signature
                )
            logger.info("Model logged successfully")

            run_id = run.info.run_id
            model_name='model'
            logger.info(f"Run ID: {run_id}, Model name: {model_name}")

            return run_id, model_name

    except Exception:
        logger.exception("Error occurred during MLflow logging")
        raise


def save_ids(run_id: str, model_name: str):
    try:
        logger.info("Saving run_id and model_id to JSON")

        os.makedirs('reports', exist_ok=True)

        ids = {
            'model_name': model_name,
            'run_id': run_id
        }

        file_path = os.path.join('reports', 'data.json')

        with open(file_path, "w") as f:
            json.dump(ids, f, indent=4)

        logger.info(f"IDs saved successfully at {file_path}")

    except Exception:
        logger.exception("Error occurred while saving IDs")
        raise


def main() -> None:
    try:
        logger.info("Training pipeline started")

        # Load Data
        data_path = os.path.join('data', 'processed')
        train_ds = load_data(data_path)

        # Split features & target
        x_train = train_ds.drop('PlacementStatus', axis=1)
        y_train = train_ds['PlacementStatus']

        logger.info("Feature-target split completed")

        # Model parameters
        model_params = {
            'learning_rate': 0.1,
            'max_depth': 3,
            'min_samples_split': 5,
            'n_estimators': 100,
            'subsample': 0.5
        }

        # Signature
        model_signature = infer_signature(x_train, y_train)
        logger.info("Model signature created")

        # Train model
        model = train_model(x_train, y_train, model_params)

        save_model_pickle(model, "gradient_boosting_model")


        # Log model
        run_id, model_name = log_model_and_parameters(
            model,
            model_params,
            model_signature
        )

        # Save IDs
        save_ids(run_id, model_name)

        logger.info("Training pipeline completed successfully")

    except Exception:
        logger.exception("Pipeline failed")
        raise


if __name__ == "__main__":
    main()