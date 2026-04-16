"""
Stage 3 — Model Training
- Trains XGBoost classifier on SMOTE-balanced data
- Logs ALL parameters, metrics, and the model artifact to MLflow
- Saves model locally for evaluate.py and deployment
"""

import os
import json
import logging
import yaml
import joblib
import pandas as pd
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [TRAIN] %(message)s")
log = logging.getLogger(__name__)


def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train(params: dict) -> None:
    processed_dir = params["data"]["processed_dir"]
    mp = params["model"]
    mlp = params["mlflow"]

    log.info("Loading training data...")
    X_train = pd.read_csv(f"{processed_dir}X_train_scaled.csv")
    y_train = pd.read_csv(f"{processed_dir}y_train_resampled.csv").squeeze()
    X_test = pd.read_csv(f"{processed_dir}X_test_scaled.csv")
    y_test = pd.read_csv(f"{processed_dir}y_test.csv").squeeze()

    # Compute scale_pos_weight from original (pre-SMOTE) ratio
    neg = (y_test == 0).sum()
    pos = (y_test == 1).sum()
    scale_pos_weight = neg / pos
    log.info(f"scale_pos_weight (from test): {scale_pos_weight:.2f}")

    # Configure MLflow
    mlflow.set_tracking_uri(mlp["tracking_uri"])
    mlflow.set_experiment(mlp["experiment_name"])

    with mlflow.start_run(run_name="xgboost_fraud_v1") as run:
        log.info(f"MLflow run ID: {run.info.run_id}")

        # Log all hyperparameters
        mlflow.log_params({
            "n_estimators": mp["n_estimators"],
            "max_depth": mp["max_depth"],
            "learning_rate": mp["learning_rate"],
            "subsample": mp["subsample"],
            "colsample_bytree": mp["colsample_bytree"],
            "scale_pos_weight": round(scale_pos_weight, 2),
            "eval_metric": mp["eval_metric"],
            "random_state": mp["random_state"],
            "smote": True,
            "dataset": "creditcard.csv",
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        })

        # Build and train model
        model = XGBClassifier(
            n_estimators=mp["n_estimators"],
            max_depth=mp["max_depth"],
            learning_rate=mp["learning_rate"],
            subsample=mp["subsample"],
            colsample_bytree=mp["colsample_bytree"],
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric=mp["eval_metric"],
            random_state=mp["random_state"],
        )

        log.info("Training XGBoost model...")
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=50
        )

        # Quick evaluation on test set
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        auc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)

        log.info(f"AUC-ROC: {auc:.4f} | F1: {f1:.4f}")

        # Log metrics to MLflow
        mlflow.log_metrics({"auc_roc": auc, "f1_score": f1})

        # Log model artifact to MLflow
        mlflow.xgboost.log_model(model, artifact_path="model")

        # Save locally for downstream stages
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/fraud_model.pkl")
        log.info("Model saved to models/fraud_model.pkl")

        # Save metrics to reports (for DVC metrics tracking)
        os.makedirs("reports", exist_ok=True)
        metrics = {"auc_roc": round(auc, 4), "f1_score": round(f1, 4)}
        with open("reports/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        mlflow.log_artifact("reports/metrics.json")
        log.info(f"Training complete. Run ID: {run.info.run_id}")


if __name__ == "__main__":
    params = load_params()
    train(params)
