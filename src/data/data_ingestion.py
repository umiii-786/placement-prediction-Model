import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.model_selection import train_test_split
import yaml
import os
import pandas as pd
from src.logging_config import logger


def load_data(file_name: str) -> pd.DataFrame:
    try:
        logger.info("Starting data loading from Kaggle")

        df = kagglehub.dataset_load(
            KaggleDatasetAdapter.PANDAS,
            "ruchikakumbhar/placement-prediction-dataset",
            file_name
        )

        logger.info(f"Data loaded successfully with shape: {df.shape}")
        return df

    except Exception as e:
        logger.exception("Error occurred while loading data")
        raise e


def load_yaml(yaml_path: str) -> float:
    try:
        logger.info(f"Loading YAML config from: {yaml_path}")

        with open(yaml_path, 'r') as file:
            params = yaml.safe_load(file)

        test_size = params['data_ingestion']['test_size']
        logger.info(f"Test size loaded: {test_size}")

        return test_size

    except Exception as e:
        logger.exception("Error occurred while loading YAML file")
        raise e


def save_data(save_path: str, train_ds: pd.DataFrame, test_ds: pd.DataFrame):
    try:
        logger.info(f"Saving datasets to path: {save_path}")

        os.makedirs(save_path, exist_ok=True)

        train_path = os.path.join(save_path, 'train_ds.csv')
        test_path = os.path.join(save_path, 'test_ds.csv')

        train_ds.to_csv(train_path, index=False)
        test_ds.to_csv(test_path, index=False)

        logger.info("Train and Test datasets saved successfully")

    except Exception as e:
        logger.exception("Error occurred while saving data")
        raise e


def main() -> None:
    try:
        logger.info("Pipeline execution started")

        file_name = "placementdata.csv"
        save_path = os.path.join('data', 'raw')

        # Load Data
        df = load_data(file_name=file_name)
        test_size = load_yaml(yaml_path="params.yaml")

        # Split Data
        logger.info("Splitting data into train and test")
        train_ds, test_ds = train_test_split(
            df,
            test_size=0.25,
            shuffle=True,
            random_state=404
        )

        logger.info(f"Train shape: {train_ds.shape}, Test shape: {test_ds.shape}")

        # Save Data
        save_data(save_path, train_ds, test_ds)

        logger.info("Pipeline execution completed successfully")


    except Exception as e:
        logger.exception("Pipeline failed")
        raise e


if __name__ == "__main__":
    main()