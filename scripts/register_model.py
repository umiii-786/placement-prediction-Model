import mlflow
from mlflow.tracking import MlflowClient
from src.logging_config import logger
import json
import os



dagshub_pat=os.getenv("DAGSHUB_PAT")

if not dagshub_pat:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_pat
os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_pat

mlflow.set_tracking_uri("https://dagshub.com/umiii-786/placement-prediction-Model.mlflow")


def register_model_new(model_id: str, model_name: str,reg_model_name):
    try:
        logger.info("Starting model registration (MLflow)")
        model_uri = f"models:/{model_id}"

        client = MlflowClient()

        # Step 1:  register the model 
        mv=mlflow.register_model(
            model_uri,
            reg_model_name
        )
        logger.info(f"Registered model '{model_name}'  with version {mv.version}")


        return mv.name, mv.version


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


    reg_model_name = "placement_prediction_model"

    # Register model
    name, version = register_model_new(
        model_id=ids['model_id'],
        model_name=ids['model_name'],
        reg_model_name=reg_model_name
    )
    print(name,' ',version )

    print('in the main',name,version)
    # Promote to production
    promote_model_to_production(name, version)

    print(f"Model '{name}' version {version} is now in PRODUCTION")

# ----------------- Entry Point -----------------
if __name__ == "__main__":
    main()