"""
Stage 5 — Model Registration
- Promotes the trained model to MLflow Model Registry
- Tags it as 'Staging' for governance review
- Only registers if AUC-ROC passes the minimum threshold
- This demonstrates ML governance: no model goes to production without approval
"""

import json
import logging
import sys
import yaml
import mlflow
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [REGISTER] %(message)s")
log = logging.getLogger(__name__)


def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def register(params: dict) -> None:
    mlp = params["mlflow"]
    model_name = mlp["model_name"]
    min_auc = params["evaluate"]["min_auc_roc"]

    # Load evaluation metrics
    with open("reports/eval_metrics.json", "r") as f:
        metrics = json.load(f)

    auc = metrics["auc_roc"]
    log.info(f"Current model AUC-ROC: {auc}")

    if auc < min_auc:
        log.error(f"Model does not meet quality threshold ({auc:.4f} < {min_auc}). Skipping registration.")
        sys.exit(1)

    mlflow.set_tracking_uri(mlp["tracking_uri"])
    client = MlflowClient()

    # Find the most recent training run that has a logged model artifact
    experiment = client.get_experiment_by_name(mlp["experiment_name"])
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.mlflow.runName = 'xgboost_fraud_v1'",
        order_by=["start_time DESC"],
        max_results=1
    )

    if not runs:
        # Fallback: find any run with a logged model
        all_runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=20
        )
        runs = [r for r in all_runs if r.data.tags.get("mlflow.log-model.history")]
        if not runs:
            log.error("No MLflow training runs with logged models found. Run train.py first.")
            sys.exit(1)

    run_id = runs[0].info.run_id
    model_uri = f"runs:/{run_id}/model"

    log.info(f"Registering model from run: {run_id}")

    # Register model
    result = mlflow.register_model(model_uri=model_uri, name=model_name)
    version = result.version

    log.info(f"Model registered: {model_name} version {version}")

    # Tag as Staging (governance step — manual promotion to Production required)
    client.set_model_version_tag(model_name, version, "auc_roc", str(auc))
    client.set_model_version_tag(model_name, version, "dataset", "creditcard.csv")
    client.set_model_version_tag(model_name, version, "approved_by", "pending_review")

    # Transition to Staging
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Staging"
    )

    log.info(f"Model {model_name} v{version} promoted to Staging.")
    log.info("Governance note: Promote to Production manually after human review.")


if __name__ == "__main__":
    params = load_params()
    register(params)
