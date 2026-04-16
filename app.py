"""
FastAPI Inference Application
Serves the trained XGBoost fraud detection model as a REST API.
Includes: prediction endpoint, health check, monitoring metrics, logging.
"""

import os
import uuid
import time
import logging
from datetime import datetime
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── Logging setup ─────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [API] %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/predictions.log"),
    ]
)
log = logging.getLogger(__name__)

# ── Load model and scaler ─────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "models/fraud_model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.pkl")
PARAMS_PATH = "params.yaml"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(PARAMS_PATH) as f:
        params = yaml.safe_load(f)
    THRESHOLD = params["evaluate"]["threshold"]
    MODEL_VERSION = os.getenv("MODEL_VERSION", "1")
    log.info(f"Model loaded from {MODEL_PATH} | threshold={THRESHOLD}")
except Exception as e:
    log.error(f"Failed to load model: {e}")
    model, scaler = None, None
    THRESHOLD = 0.5
    MODEL_VERSION = "unknown"

# ── In-memory monitoring counters ─────────────────────────────────
stats = {
    "total_predictions": 0,
    "fraud_predictions": 0,
    "total_latency_ms": 0.0,
    "errors": 0,
}

# ── FastAPI app ───────────────────────────────────────────────────
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit card fraud detection powered by XGBoost + MLOps pipeline",
    version="1.0.0",
)


# ── Request schema ────────────────────────────────────────────────
class Transaction(BaseModel):
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float = Field(..., ge=0, description="Transaction amount in USD")
    Time: Optional[float] = Field(default=0.0, description="Seconds since first transaction")


# ── Response schema ───────────────────────────────────────────────
class PredictionResponse(BaseModel):
    transaction_id: str
    prediction: int
    fraud_probability: float
    risk_level: str
    model_version: str
    timestamp: str
    latency_ms: float


# ── Endpoints ─────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "model_version": MODEL_VERSION,
        "threshold": THRESHOLD,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/metrics")
def metrics():
    avg_latency = (
        stats["total_latency_ms"] / stats["total_predictions"]
        if stats["total_predictions"] > 0 else 0
    )
    fraud_rate = (
        stats["fraud_predictions"] / stats["total_predictions"]
        if stats["total_predictions"] > 0 else 0
    )
    return {
        "total_predictions": stats["total_predictions"],
        "fraud_predictions": stats["fraud_predictions"],
        "fraud_rate": round(fraud_rate, 4),
        "average_latency_ms": round(avg_latency, 2),
        "errors": stats["errors"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    if model is None:
        stats["errors"] += 1
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()
    txn_id = f"txn_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    try:
        # Build feature dataframe
        features = transaction.dict()
        df = pd.DataFrame([features])

        # Scale Amount and Time (same as training)
        df[["Amount", "Time"]] = scaler.transform(df[["Amount", "Time"]])

        # Predict
        prob = float(model.predict_proba(df)[:, 1][0])
        pred = int(prob >= THRESHOLD)

        # Risk level labeling
        if prob >= 0.80:
            risk = "HIGH"
        elif prob >= 0.50:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        latency = round((time.time() - start) * 1000, 2)

        # Update monitoring stats
        stats["total_predictions"] += 1
        stats["total_latency_ms"] += latency
        if pred == 1:
            stats["fraud_predictions"] += 1

        # Log every prediction
        log.info(
            f"txn_id={txn_id} | prediction={pred} | probability={prob:.4f} | "
            f"risk={risk} | amount={transaction.Amount:.2f} | latency={latency}ms"
        )

        return PredictionResponse(
            transaction_id=txn_id,
            prediction=pred,
            fraud_probability=round(prob, 4),
            risk_level=risk,
            model_version=MODEL_VERSION,
            timestamp=datetime.utcnow().isoformat(),
            latency_ms=latency,
        )

    except Exception as e:
        stats["errors"] += 1
        log.error(f"Prediction error for {txn_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {
        "service": "Fraud Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
    }
