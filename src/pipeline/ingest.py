"""
Stage 1 — Data Ingestion
Loads raw creditcard.csv, performs train/test split, saves to data/processed/.
All parameters are read from params.yaml so DVC can track changes.
"""

import os
import logging
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [INGEST] %(message)s")
log = logging.getLogger(__name__)


def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ingest(params: dict) -> None:
    raw_path = params["data"]["raw_path"]
    processed_dir = params["data"]["processed_dir"]
    test_size = params["data"]["test_size"]
    random_state = params["data"]["random_state"]

    os.makedirs(processed_dir, exist_ok=True)

    log.info(f"Loading raw data from {raw_path}")
    df = pd.read_csv(raw_path)

    log.info(f"Dataset shape: {df.shape}")
    log.info(f"Fraud rate: {df['Class'].mean():.4%}")

    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y          # preserve fraud ratio in both splits
    )

    log.info(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
    log.info(f"Train fraud cases: {y_train.sum()} | Test fraud cases: {y_test.sum()}")

    X_train.to_csv(f"{processed_dir}X_train.csv", index=False)
    X_test.to_csv(f"{processed_dir}X_test.csv", index=False)
    y_train.to_csv(f"{processed_dir}y_train.csv", index=False)
    y_test.to_csv(f"{processed_dir}y_test.csv", index=False)

    log.info("Ingestion complete. Files saved to data/processed/")


if __name__ == "__main__":
    params = load_params()
    ingest(params)
