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
    """
    Register a model in MLflow. If the model already exists, fetch the latest version.
    """
    try:
        logger.info("Starting model registration (NEW MLflow API)")

        model_uri = f"runs:/{run_id}/model"
        client = MlflowClient()

        try:
            # Try to register the model
            result = mlflow.register_model(model_uri=model_uri, name=model_name)
            logger.info(f"Registered model '{model_name}' created")
        except Exception as e:
            logger.warning(f"Model '{model_name}' might already exist. Fetching latest version.")
            # Fetch latest version safely
            all_versions = client.get_latest_versions(name=model_name)
            if not all_versions:
                raise ValueError(f"No versions found for existing model '{model_name}'")
            result = all_versions[0]

        logger.info(f"Model URI: {model_uri}")
        logger.info(f"Model registered with version {result.version}")

        return model_name, result.version

    except Exception:
        logger.exception("Error occurred during model registration")
        raise


def promote_model_to_production(model_name: str, version: int):
    """
    Promote a registered model version to Production stage using alias.
    """
    try:
        logger.info(f"Promoting model {model_name} v{version} to Production")

        client = MlflowClient()

        # Modern MLflow alias method
        client.set_registered_model_alias(
            name=model_name,
            alias="production",
            version=str(version)  # ensure version is string
        )

        logger.info(f"Model '{model_name}' v{version} promoted to Production")

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

    model_name = "placement-prediction-model-GB"

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