import pandas as pd
import os
from src.logging_config import logger
from sklearn.preprocessing import LabelEncoder
import pickle


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


def transform_input_features(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.info("Transforming input features")

        df = df.copy()  # avoid inplace issues

        df.drop('StudentID', axis=1, inplace=True)

        df['ExtracurricularActivities'] = df['ExtracurricularActivities'].map({'No': 0, 'Yes': 1})
        df['PlacementTraining'] = df['PlacementTraining'].map({'No': 0, 'Yes': 1})

        logger.debug("Input feature transformation completed")

        return df

    except Exception as e:
        logger.exception("Error occurred while transforming input features")
        raise e


def transform_target_features(train_ds: pd.DataFrame, test_ds: pd.DataFrame) -> tuple:
    try:
        logger.info("Encoding target variable")

        lbl = LabelEncoder()
        lbl.fit(train_ds['PlacementStatus'])

        train_ds['PlacementStatus'] = lbl.transform(train_ds['PlacementStatus'])
        test_ds['PlacementStatus'] = lbl.transform(test_ds['PlacementStatus'])

        logger.info("Target encoding completed")

        return lbl, train_ds, test_ds

    except Exception as e:
        logger.exception("Error occurred while transforming target features")
        raise e


def save_target_encoder(lbl: LabelEncoder):
    try:
        logger.info("Saving target encoder")

        with open(os.path.join('models','target_encoder.pkl'), 'wb') as f:
            pickle.dump(lbl, f)

        logger.info("Target encoder saved successfully")

    except Exception as e:
        logger.exception("Error occurred while saving target encoder")
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
        logger.info("Feature Engineering Pipeline started")

        data_path = os.path.join('data', 'interim')
        save_path = os.path.join('data', 'processed')

        train_ds, test_ds = load_data(data_path)

        train_ds = transform_input_features(train_ds)
        test_ds = transform_input_features(test_ds)

        logger.info("Removing duplicates from training data")
        train_ds = train_ds.drop_duplicates()

        lbl, train_ds, test_ds = transform_target_features(train_ds, test_ds)

        save_target_encoder(lbl)

        save_data(save_path, train_ds, test_ds)

        logger.info("Feature Engineering Pipeline completed successfully")

    except Exception as e:
        logger.exception("Pipeline failed")
        raise e


if __name__ == '__main__':
    main()