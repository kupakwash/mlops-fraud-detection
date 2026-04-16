"""
Stage 2 — Preprocessing
- Scales Amount and Time features (V1-V28 are already PCA-scaled)
- Applies SMOTE to handle severe class imbalance (0.17% fraud)
- Saves scaled arrays and fitted scaler for inference use
"""

import os
import logging
import yaml
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [PREPROCESS] %(message)s")
log = logging.getLogger(__name__)


def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def preprocess(params: dict) -> None:
    processed_dir = params["data"]["processed_dir"]
    scale_features = params["preprocess"]["scale_features"]
    smote_state = params["preprocess"]["smote_random_state"]

    log.info("Loading processed split data...")
    X_train = pd.read_csv(f"{processed_dir}X_train.csv")
    X_test = pd.read_csv(f"{processed_dir}X_test.csv")
    y_train = pd.read_csv(f"{processed_dir}y_train.csv").squeeze()

    # Scale Amount and Time
    log.info(f"Scaling features: {scale_features}")
    scaler = StandardScaler()
    X_train[scale_features] = scaler.fit_transform(X_train[scale_features])
    X_test[scale_features] = scaler.transform(X_test[scale_features])

    # Save scaler for inference pipeline
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    log.info("Scaler saved to models/scaler.pkl")

    # Apply SMOTE to training data only (never to test data — data leakage risk)
    log.info(f"Class distribution before SMOTE: {y_train.value_counts().to_dict()}")
    smote = SMOTE(random_state=smote_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    log.info(f"Class distribution after SMOTE: {pd.Series(y_resampled).value_counts().to_dict()}")

    # Save outputs
    pd.DataFrame(X_resampled, columns=X_train.columns).to_csv(
        f"{processed_dir}X_train_scaled.csv", index=False
    )
    X_test.to_csv(f"{processed_dir}X_test_scaled.csv", index=False)
    pd.Series(y_resampled, name="Class").to_csv(
        f"{processed_dir}y_train_resampled.csv", index=False
    )

    log.info("Preprocessing complete.")


if __name__ == "__main__":
    params = load_params()
    preprocess(params)
