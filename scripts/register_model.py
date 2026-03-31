import mlflow
from mlflow.tracking import MlflowClient
from src.logging_config import logger
import json
import os

# ----------------- DagsHub Authentication -----------------
dagshub_pat = "a55ae4d7356bf84fa662753c4cff9084c43da67d"
if not dagshub_pat:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_pat
os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_pat

mlflow.set_tracking_uri("https://dagshub.com/umiii-786/placement-prediction-Model.mlflow")

# ----------------- Register Model -----------------
def register_model_new(run_id: str, model_name: str):
    """
    Register a model in MLflow. If it already exists, fetch the latest version.
    """
    try:
        logger.info("Starting model registration (MLflow)")

        model_uri = f"runs:/{run_id}/model"
        client = MlflowClient()

        try:
            # Try registering the model
            result = mlflow.register_model(model_uri=model_uri, name=model_name)
            print(result)
            logger.info(f"Registered model '{model_name}' successfully")
        except Exception as e:
            logger.warning(f"Model '{model_name}' might already exist: {e}")
            # Fetch all versions safely
            all_versions = client.get_latest_versions(name=model_name)
            
            if not all_versions or len(all_versions)==0:
                return model_name,1
          
            result = all_versions[0]  # pick the latest version

        logger.info(f"Model URI: {model_uri}")
        logger.info(f"Model registered with version: {result}")

        return model_name, result

    except Exception:
        logger.exception("Error occurred during model registration")
        raise

# ----------------- Promote to Production -----------------
def promote_model_to_production(model_name: str, version):
    """
    Promote a registered model version to Production stage using alias.
    """
    try:
        client = MlflowClient()

        logger.info(f"Promoting model '{model_name}' v{version} to Production")

        # Use alias 'production' (modern MLflow)
        client.set_registered_model_alias(
            name=model_name,
            alias="production",
            version=str(version)  # ensure version is string
        )

        logger.info(f"Model '{model_name}' v{version} promoted to Production")

    except Exception:
        logger.exception("Error promoting model")
        raise

# ----------------- Load run IDs from JSON -----------------
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

# ----------------- Main -----------------
def main() -> None:
    ids = get_ids("reports/data.json")
    model_name = "placement-prediction-model-GB"

    # Register model
    name, version = register_model_new(
        run_id=ids['run_id'],
        model_name=model_name
    )

    print('in the main',name,version)
    # Promote to production
    promote_model_to_production(name, version)

    print(f"Model '{name}' version {version} is now in PRODUCTION")

# ----------------- Entry Point -----------------
if __name__ == "__main__":
    main()