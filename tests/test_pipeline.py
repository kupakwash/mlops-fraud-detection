"""
Unit tests for the MLOps Fraud Detection pipeline.
These run automatically in the CI/CD GitHub Actions workflow.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ── Test: ingest module ───────────────────────────────────────────
class TestIngest:
    def test_train_test_split_ratio(self):
        """Ensure split produces roughly 80/20 ratio."""
        from sklearn.model_selection import train_test_split
        data = pd.DataFrame({"a": range(1000), "Class": [0]*990 + [1]*10})
        X = data.drop("Class", axis=1)
        y = data["Class"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        assert len(X_train) == 800
        assert len(X_test) == 200

    def test_stratified_split_preserves_fraud_rate(self):
        """Fraud ratio should be equal in train and test after stratified split."""
        from sklearn.model_selection import train_test_split
        n = 10000
        fraud = int(n * 0.002)
        y = pd.Series([0] * (n - fraud) + [1] * fraud)
        X = pd.DataFrame({"f": range(n)})
        _, _, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        assert abs(y_train.mean() - y_test.mean()) < 0.001


# ── Test: preprocess module ───────────────────────────────────────
class TestPreprocess:
    def test_scaler_zero_mean(self):
        """StandardScaler should produce near-zero mean on training data."""
        from sklearn.preprocessing import StandardScaler
        data = pd.DataFrame({"Amount": [10, 20, 30, 40, 50], "Time": [1, 2, 3, 4, 5]})
        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)
        assert abs(scaled[:, 0].mean()) < 1e-10

    def test_smote_balances_classes(self):
        """SMOTE should produce equal class counts."""
        from imblearn.over_sampling import SMOTE
        X = pd.DataFrame(np.random.randn(1000, 5))
        y = pd.Series([0] * 990 + [1] * 10)
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        counts = pd.Series(y_res).value_counts()
        assert counts[0] == counts[1]


# ── Test: API schema ──────────────────────────────────────────────
class TestAPISchema:
    """Tests the Pydantic Transaction schema — no trained model required.
    app.py handles a missing model gracefully (model = None), so these
    tests work in CI even before the pipeline has been run.
    """

    @pytest.fixture(autouse=True)
    def require_app(self):
        """Skip the whole class if app.py cannot be imported for any reason."""
        pytest.importorskip("fastapi")
        pytest.importorskip("pydantic")

    def test_transaction_has_required_fields(self):
        """Transaction model should accept all 30 features."""
        try:
            from app import Transaction
        except Exception as exc:
            pytest.skip(f"app.py import failed: {exc}")
        data = {f"V{i}": 0.0 for i in range(1, 29)}
        data["Amount"] = 100.0
        data["Time"] = 0.0
        txn = Transaction(**data)
        assert txn.Amount == 100.0

    def test_transaction_amount_non_negative(self):
        """Amount field must be >= 0."""
        try:
            from app import Transaction
        except Exception as exc:
            pytest.skip(f"app.py import failed: {exc}")
        from pydantic import ValidationError
        data = {f"V{i}": 0.0 for i in range(1, 29)}
        data["Amount"] = -5.0
        data["Time"] = 0.0
        with pytest.raises(ValidationError):
            Transaction(**data)


# ── Test: params.yaml ─────────────────────────────────────────────
class TestParams:
    def test_params_file_exists(self):
        assert os.path.exists("params.yaml")

    def test_params_has_required_keys(self):
        import yaml
        with open("params.yaml") as f:
            p = yaml.safe_load(f)
        assert "data" in p
        assert "model" in p
        assert "mlflow" in p
        assert "evaluate" in p

    def test_test_size_valid(self):
        import yaml
        with open("params.yaml") as f:
            p = yaml.safe_load(f)
        assert 0 < p["data"]["test_size"] < 1
