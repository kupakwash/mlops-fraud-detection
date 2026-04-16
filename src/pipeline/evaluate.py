"""
Stage 4 — Model Evaluation
- Loads trained model and test data
- Computes full evaluation suite: AUC-ROC, F1, Precision, Recall, Confusion Matrix
- Saves all metrics to reports/ for DVC metrics tracking and MLflow logging
- Acts as CI/CD gate: exits with error if AUC-ROC is below threshold
"""

import json
import logging
import sys

import joblib
import mlflow
import pandas as pd
import yaml
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, confusion_matrix, classification_report,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [EVALUATE] %(message)s")
log = logging.getLogger(__name__)


def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def evaluate(params: dict) -> None:
    processed_dir = params["data"]["processed_dir"]
    threshold = params["evaluate"]["threshold"]
    min_auc = params["evaluate"]["min_auc_roc"]
    mlp = params["mlflow"]

    log.info("Loading model and test data...")
    model = joblib.load("models/fraud_model.pkl")
    X_test = pd.read_csv(f"{processed_dir}X_test_scaled.csv")
    y_test = pd.read_csv(f"{processed_dir}y_test.csv").squeeze()

    # Predictions
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    # Metrics
    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()

    log.info("=" * 50)
    log.info(f"AUC-ROC   : {auc:.4f}")
    log.info(f"F1 Score  : {f1:.4f}")
    log.info(f"Precision : {precision:.4f}")
    log.info(f"Recall    : {recall:.4f}")
    log.info("Confusion Matrix:")
    log.info(f"  TN={cm[0][0]}  FP={cm[0][1]}")
    log.info(f"  FN={cm[1][0]}  TP={cm[1][1]}")
    log.info("=" * 50)
    log.info("\n" + classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    # Save full metrics
    eval_metrics = {
        "auc_roc": round(auc, 4),
        "f1_score": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "threshold": threshold,
    }
    with open("reports/eval_metrics.json", "w") as f:
        json.dump(eval_metrics, f, indent=2)

    with open("reports/confusion_matrix.json", "w") as f:
        json.dump({"confusion_matrix": cm}, f, indent=2)

    # Log to MLflow (resume most recent run)
    mlflow.set_tracking_uri(mlp["tracking_uri"])
    mlflow.set_experiment(mlp["experiment_name"])
    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_metrics(eval_metrics)
        mlflow.log_artifact("reports/eval_metrics.json")
        mlflow.log_artifact("reports/confusion_matrix.json")

    # CI/CD Gate — hard fail if below minimum AUC threshold
    if auc < min_auc:
        log.error(f"QUALITY GATE FAILED: AUC-ROC {auc:.4f} < threshold {min_auc}")
        sys.exit(1)

    log.info(f"Quality gate PASSED: AUC-ROC {auc:.4f} >= {min_auc}")


if __name__ == "__main__":
    params = load_params()
    evaluate(params)
