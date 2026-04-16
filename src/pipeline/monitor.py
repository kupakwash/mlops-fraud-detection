"""
Monitoring — Data Drift Detection
Compares incoming inference data distribution against the training baseline.
Run this script periodically (e.g., daily cron job or GitHub Actions schedule).
Uses statistical tests to flag feature drift before it degrades model performance.
"""

import os
import json
import logging
import yaml
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [DRIFT] %(message)s")
log = logging.getLogger(__name__)


def load_params(path: str = "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def compute_baseline_stats(df: pd.DataFrame) -> dict:
    """Compute mean and std for each feature from training data."""
    stats_dict = {}
    for col in df.columns:
        stats_dict[col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
        }
    return stats_dict


def detect_drift(baseline_df: pd.DataFrame, current_df: pd.DataFrame,
                 threshold_p: float = 0.05) -> dict:
    """
    Run Kolmogorov-Smirnov test for each feature.
    p-value < threshold means the distributions are significantly different (drift detected).
    """
    results = {}
    drifted_features = []

    for col in baseline_df.columns:
        if col not in current_df.columns:
            continue
        ks_stat, p_value = stats.ks_2samp(baseline_df[col], current_df[col])
        drifted = bool(p_value < threshold_p)
        results[col] = {
            "ks_statistic": round(float(ks_stat), 4),
            "p_value": round(float(p_value), 4),
            "drift_detected": drifted,
        }
        if drifted:
            drifted_features.append(col)

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "threshold_p_value": threshold_p,
        "total_features_checked": len(results),
        "drifted_features_count": len(drifted_features),
        "drifted_features": drifted_features,
        "drift_detected": len(drifted_features) > 0,
        "feature_results": results,
    }


def run_drift_check(params: dict) -> None:
    processed_dir = params["data"]["processed_dir"]

    # Load training baseline
    log.info("Loading training baseline...")
    X_train = pd.read_csv(f"{processed_dir}X_train_scaled.csv")
    baseline_stats = compute_baseline_stats(X_train)

    # Save baseline stats for reference
    os.makedirs("reports", exist_ok=True)
    with open("reports/baseline_stats.json", "w") as f:
        json.dump(baseline_stats, f, indent=2)
    log.info("Baseline stats saved to reports/baseline_stats.json")

    # Simulate current inference data (in production this would be a live log)
    # Here we add synthetic drift to Amount for demonstration
    log.info("Simulating current inference window data...")
    current_data = X_train.sample(n=min(5000, len(X_train)), random_state=99).copy()
    # Inject drift: shift Amount by 2 standard deviations
    current_data["Amount"] = current_data["Amount"] + 2.0

    # Run drift detection
    log.info("Running Kolmogorov-Smirnov drift tests...")
    drift_report = detect_drift(X_train, current_data, threshold_p=0.05)

    # Save drift report
    with open("reports/drift_report.json", "w") as f:
        json.dump(drift_report, f, indent=2)

    # Summary logging
    log.info("=" * 55)
    log.info(f"Drift check complete: {drift_report['timestamp']}")
    log.info(f"Features checked : {drift_report['total_features_checked']}")
    log.info(f"Drifted features : {drift_report['drifted_features_count']}")
    if drift_report["drift_detected"]:
        log.warning(f"DRIFT DETECTED in: {drift_report['drifted_features']}")
        log.warning("ACTION REQUIRED: Consider triggering model retraining.")
    else:
        log.info("No significant drift detected. Model is stable.")
    log.info("=" * 55)
    log.info("Full report saved to reports/drift_report.json")


if __name__ == "__main__":
    params = load_params()
    run_drift_check(params)
