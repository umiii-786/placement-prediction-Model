import pandas as pd
import os
from src.logging_config import logger


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

    except Exception as e:
        logger.exception("Error occurred while loading data")
        raise e


def iqr_cap(series: pd.Series) -> pd.Series:
    try:
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        return series.clip(lower, upper)

    except Exception as e:
        logger.exception("Error in IQR capping")
        raise e


def handle_outliers(train_ds: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.info("Handling outliers using IQR method")

        train_ds['SoftSkillsRating'] = train_ds.groupby(
            'PlacementTraining', group_keys=False
        )['SoftSkillsRating'].transform(iqr_cap)

        train_ds['SSC_Marks'] = train_ds.groupby(
            'PlacementStatus', group_keys=False
        )['SSC_Marks'].transform(iqr_cap)

        train_ds['HSC_Marks'] = train_ds.groupby(
            'PlacementStatus', group_keys=False
        )['HSC_Marks'].transform(iqr_cap)

        logger.info("Outliers handled successfully")

        return train_ds

    except Exception as e:
        logger.exception("Error occurred while handling outliers")
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
        logger.info("Pipeline started")

        data_path = os.path.join('data', 'raw')
        save_path = os.path.join('data', 'interim')

        train_ds, test_ds = load_data(data_path)

        train_ds = handle_outliers(train_ds)

        save_data(save_path, train_ds, test_ds)

        logger.info("Pipeline completed successfully")

    except Exception as e:
        logger.exception("Pipeline failed")
        raise e


if __name__ == '__main__':
    main()