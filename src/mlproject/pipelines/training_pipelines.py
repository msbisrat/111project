# src/mlproject/pipelines/training_pipelines.py

import os
import sys

# Ensure project root is on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.mlproject.components.data_transformation import DataTransformation
from src.mlproject.components.model_trainer import ModelTrainer
from src.mlproject.logger import logging
from src.mlproject.exception import CustomException

import pandas as pd
from sklearn.model_selection import train_test_split


DATA_PATH = os.path.join(PROJECT_ROOT, "notebook", "data", "cleaned_data.csv")
TRAIN_PATH = os.path.join(PROJECT_ROOT, "artifacts", "train.csv")
TEST_PATH  = os.path.join(PROJECT_ROOT, "artifacts", "test.csv")


def run_training_pipeline():
    try:
        # ── Step 1: Load and split data ───────────────────────────────────────
        logging.info(f"Loading data from {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
        logging.info(f"Dataset shape: {df.shape}")

        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

        os.makedirs(os.path.join(PROJECT_ROOT, "artifacts"), exist_ok=True)
        train_df.to_csv(TRAIN_PATH, index=False)
        test_df.to_csv(TEST_PATH,  index=False)
        logging.info(f"Train/test split saved to artifacts/")

        # ── Step 2: Data transformation ───────────────────────────────────────
        logging.info("Starting data transformation...")
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
            TRAIN_PATH, TEST_PATH
        )
        logging.info(f"Preprocessor saved to {preprocessor_path}")

        # ── Step 3: Model training ────────────────────────────────────────────
        logging.info("Starting model training...")
        model_trainer = ModelTrainer()
        accuracy = model_trainer.initiate_model_trainer(train_arr, test_arr)
        logging.info(f"Training complete. Best model accuracy: {accuracy:.4f}")
        print(f"\n✅ Training complete. Best model accuracy: {accuracy:.4f}")
        print(f"✅ Preprocessor saved to: {preprocessor_path}")
        print(f"✅ Model saved to: artifacts/model.pkl")

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_training_pipeline()