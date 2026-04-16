"""
Smoke tests — fast, no model or dataset required.
These run in CI to verify project structure and configuration before
the full pipeline executes.
"""

import os
import yaml
import pytest


class TestProjectStructure:
    """Verify all required files exist in the repo."""

    def test_params_yaml_exists(self):
        assert os.path.exists("params.yaml"), "params.yaml is missing"

    def test_dvc_yaml_exists(self):
        assert os.path.exists("dvc.yaml"), "dvc.yaml is missing"

    def test_requirements_txt_exists(self):
        assert os.path.exists("requirements.txt"), "requirements.txt is missing"

    def test_pipeline_scripts_all_present(self):
        scripts = [
            "src/pipeline/ingest.py",
            "src/pipeline/preprocess.py",
            "src/pipeline/train.py",
            "src/pipeline/evaluate.py",
            "src/pipeline/register.py",
            "src/pipeline/monitor.py",
        ]
        for script in scripts:
            assert os.path.exists(script), f"Missing pipeline script: {script}"

    def test_app_py_exists(self):
        assert os.path.exists("app.py"), "app.py (inference API) is missing"

    def test_src_init_exists(self):
        assert os.path.exists("src/__init__.py")


class TestParamsConfig:
    """Validate params.yaml has correct structure and sensible values."""

    @pytest.fixture
    def params(self):
        with open("params.yaml") as f:
            return yaml.safe_load(f)

    def test_all_top_level_keys_present(self, params):
        for key in ("data", "preprocess", "model", "evaluate", "mlflow"):
            assert key in params, f"Missing top-level key in params.yaml: {key}"

    def test_data_paths_are_strings(self, params):
        assert isinstance(params["data"]["raw_path"], str)
        assert isinstance(params["data"]["processed_dir"], str)

    def test_test_size_is_valid_fraction(self, params):
        assert 0 < params["data"]["test_size"] < 1

    def test_model_n_estimators_positive(self, params):
        assert params["model"]["n_estimators"] > 0

    def test_model_max_depth_positive(self, params):
        assert params["model"]["max_depth"] > 0

    def test_learning_rate_in_range(self, params):
        lr = params["model"]["learning_rate"]
        assert 0 < lr <= 1, f"learning_rate {lr} is outside (0, 1]"

    def test_evaluate_threshold_is_valid(self, params):
        t = params["evaluate"]["threshold"]
        assert 0 < t < 1, f"threshold {t} is outside (0, 1)"

    def test_min_auc_is_valid(self, params):
        m = params["evaluate"]["min_auc_roc"]
        assert 0 < m < 1, f"min_auc_roc {m} is outside (0, 1)"

    def test_mlflow_has_required_fields(self, params):
        for field in ("experiment_name", "model_name", "tracking_uri"):
            assert field in params["mlflow"], f"Missing mlflow.{field}"


class TestModuleImports:
    """Verify all pipeline modules import cleanly (no syntax errors)."""

    def test_ingest_imports(self):
        from src.pipeline.ingest import ingest, load_params  # noqa: F401

    def test_preprocess_imports(self):
        from src.pipeline.preprocess import preprocess, load_params  # noqa: F401

    def test_train_imports(self):
        pytest.importorskip("mlflow", reason="mlflow not installed — skipping train module import")
        from src.pipeline.train import train, load_params  # noqa: F401

    def test_evaluate_imports(self):
        pytest.importorskip("mlflow", reason="mlflow not installed — skipping evaluate module import")
        from src.pipeline.evaluate import evaluate, load_params  # noqa: F401

    def test_monitor_imports(self):
        from src.pipeline.monitor import detect_drift, run_drift_check  # noqa: F401

    def test_app_transaction_importable(self):
        from app import Transaction  # noqa: F401


class TestDriftLogic:
    """Unit test the drift detection function in isolation (no data files needed)."""

    def test_detect_drift_flags_shifted_distribution(self):
        import pandas as pd
        import numpy as np
        from src.pipeline.monitor import detect_drift

        rng = np.random.default_rng(42)
        baseline = pd.DataFrame({"V1": rng.normal(0, 1, 1000),
                                  "Amount": rng.normal(100, 20, 1000)})
        # Heavily shifted current data — should trigger drift
        current = pd.DataFrame({"V1": rng.normal(0, 1, 1000),
                                 "Amount": rng.normal(500, 20, 1000)})

        result = detect_drift(baseline, current, threshold_p=0.05)

        assert result["drift_detected"] is True
        assert "Amount" in result["drifted_features"]
        assert result["feature_results"]["V1"]["drift_detected"] is False

    def test_detect_drift_no_flag_on_same_distribution(self):
        import pandas as pd
        import numpy as np
        from src.pipeline.monitor import detect_drift

        rng = np.random.default_rng(0)
        data = {"V1": rng.normal(0, 1, 2000), "Amount": rng.normal(100, 10, 2000)}
        baseline = pd.DataFrame({k: v[:1000] for k, v in data.items()})
        current = pd.DataFrame({k: v[1000:] for k, v in data.items()})

        # Use a very strict threshold (p < 0.001) — two halves of the same
        # distribution should not produce KS p-values this low.
        result = detect_drift(baseline, current, threshold_p=0.001)

        assert result["drift_detected"] is False
        assert result["drifted_features_count"] == 0

    def test_drift_result_is_json_serializable(self):
        import json
        import pandas as pd
        import numpy as np
        from src.pipeline.monitor import detect_drift

        rng = np.random.default_rng(7)
        df = pd.DataFrame({"V1": rng.normal(0, 1, 500)})
        result = detect_drift(df, df.copy())

        # Should not raise
        json.dumps(result)
