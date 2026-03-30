import mlflow
from mlflow.tracking import MlflowClient
from src.logging_config import logger
import json
import os
import dagshub


dagshub_pat=os.getenv("DAGSHUB_PAT")
if not dagshub_pat:
    raise EnvironmentError('DAGSHUB_PAT environment variable is not setted ') 
os.environ['MLFLOW_TRACKING_USERNAME']=dagshub_pat 
os.environ['MLFLOW_TRACKING_PASSWORD']=dagshub_pat 

mlflow.set_tracking_uri("https://dagshub.com/umiii-786/placement-prediction-Model.mlflow")

def register_model_new(run_id: str, model_name: str):
    try:
        logger.info("Starting model registration (NEW MLflow API)")

        client = MlflowClient()

        # Step 1: Create registered model (if not exists)
        try:
            client.create_registered_model(model_name)
            logger.info(f"Registered model '{model_name}' created")
        except Exception:
            logger.info(f"Model '{model_name}' already exists")

        # Step 2: Define model URI
        model_uri = f"runs:/{run_id}/model"
        logger.info(f"Model URI: {model_uri}")

        # Step 3: Create model version
        model_version = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=run_id
        )

        logger.info(f"Model version created: {model_version.version}")

        return model_version.name, model_version.version

    except Exception:
        logger.exception("Error occurred during model registration")
        raise


def promote_model_to_production(model_name: str, version: int):
    try:
        logger.info(f"Promoting model {model_name} v{version} to Production")

        client = MlflowClient()

        client.set_registered_model_alias(
            name=model_name,
            alias="production",
            version=version
        )

        logger.info("Model promoted to Production using alias")

    except Exception:
        logger.exception("Error promoting model")
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


def main()->None:

    ids = get_ids("reports/data.json")

    model_name = "placement_prediction_model"

    # Register model
    name, version = register_model_new(
        run_id=ids['run_id'],
        model_name=model_name
    )

    # Promote to production
    promote_model_to_production(name, version)

    print(f"Model {name} version {version} is now in PRODUCTION")

if __name__=='__main__':
    main()