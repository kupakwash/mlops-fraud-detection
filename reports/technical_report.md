# Technical Project Report
## MLOps Fraud Detection System

**Student:** Kupakwashe T. Mapuranga  
**Assessment:** CA 4 — Mini Project (23 Marks)  
**Unit:** Unit 2 — MLOps  
**Date:** 2024  

---

## 1. Problem Definition & ML Use Case

### Business Problem
Credit card fraud costs the global financial industry over **$32 billion annually**.
Financial institutions require intelligent systems that can flag suspicious transactions
in real time — before they are processed and funds are transferred.

### ML Formulation
- **Task:** Binary classification — predict whether a transaction is fraudulent (1) or legitimate (0)
- **Dataset:** Kaggle Credit Card Fraud Detection (ULB Machine Learning Group)
  - 284,807 transactions | 492 fraud cases (0.172% positive class)
  - Features: V1–V28 (PCA-anonymized), Amount, Time
- **Algorithm:** XGBoost (Extreme Gradient Boosting)
- **Deployment Strategy:** Real-time REST API (FastAPI on Azure ACI)

### Algorithm Justification
XGBoost was selected because:
1. It is the industry standard for structured/tabular fraud detection (used by Stripe, Visa, PayPal)
2. The `scale_pos_weight` parameter natively addresses the severe class imbalance (577:1 ratio)
3. It trains in under 2 minutes on CPU — compatible with automated CI/CD retraining
4. Feature importance scores provide interpretability for regulatory compliance

### Deployment Strategy Justification
Real-time REST API was chosen over batch processing because:
- Fraud decisions must be made **within milliseconds** of transaction initiation
- Batch scoring (overnight) would allow fraudulent transactions to complete before detection
- FastAPI provides async, high-throughput serving suitable for payment volumes

---

## 2. Data Versioning & Experiment Tracking

### DVC (Data Version Control)
- DVC was initialized and configured with Azure Blob Storage as the remote backend
- The raw dataset (`data/raw/creditcard.csv`) is tracked via `dvc add`
- All processed files in `data/processed/` are tracked as DVC pipeline outputs
- Any team member can reproduce the exact dataset state with `dvc pull`

**DVC Remote Configuration:**
```
Remote: Azure Blob Storage
Container: mlops-dvc
Account: [Azure Student account]
```

### MLflow Experiment Tracking
- Every training run logs: hyperparameters, metrics, the model artifact, and metadata
- Experiment name: `fraud-detection`
- Tracked metrics: AUC-ROC, F1 Score, Precision, Recall
- Model artifact stored in MLflow model registry under name `fraud_xgboost`
- MLflow UI accessible at `http://localhost:5000` during development

**Sample MLflow Run Log:**
```
Run: xgboost_fraud_v1
Params: n_estimators=200, max_depth=6, learning_rate=0.1
Metrics: auc_roc=0.9821, f1_score=0.8734
Artifact: model/ (XGBoost model in MLflow format)
```

---

## 3. Modular Pipeline Design

The pipeline follows clean code principles with five independent, testable modules:

| Module | File | Responsibility |
|--------|------|---------------|
| Ingest | `src/pipeline/ingest.py` | Load CSV, stratified train/test split |
| Preprocess | `src/pipeline/preprocess.py` | StandardScaler + SMOTE balancing |
| Train | `src/pipeline/train.py` | XGBoost training + MLflow logging |
| Evaluate | `src/pipeline/evaluate.py` | Full metrics suite + CI/CD quality gate |
| Register | `src/pipeline/register.py` | MLflow Model Registry promotion |

**Design Principles Applied:**
- Each module reads from `params.yaml` — no hardcoded values
- Each module is independently executable (`python src/pipeline/train.py`)
- DVC orchestrates the full pipeline via `dvc repro`
- Clean separation of concerns — no module depends on another's internals

**Pipeline Orchestration (dvc.yaml):**
```
ingest → preprocess → train → evaluate → register
```
DVC tracks dependencies between stages and only reruns stages whose inputs have changed.

---

## 4. CI/CD Implementation

GitHub Actions workflow (`.github/workflows/mlops.yml`) implements a 5-stage pipeline:

### Stage 1: Lint (Code Quality)
- Tool: `flake8`
- Checks: PEP8 compliance, max line length 100
- Gate: Pipeline fails if lint errors exist

### Stage 2: Unit Tests
- Tool: `pytest`
- Tests: Data split ratios, scaler behavior, SMOTE balance, API schema validation
- Gate: Pipeline fails if any test fails

### Stage 3: ML Pipeline Execution
- Pulls data from Azure Blob via DVC
- Runs `dvc repro` (full pipeline: ingest → preprocess → train → evaluate → register)
- Gate: Pipeline fails if AUC-ROC < 0.95 (evaluate.py quality gate)

### Stage 4: Docker Build & Push
- Builds Docker image with trained model baked in
- Pushes to Azure Container Registry (ACR) with SHA tag and `latest` tag

### Stage 5: Deploy to Azure ACI
- Deploys latest Docker image to Azure Container Instances
- Runs health check against live `/health` endpoint to confirm successful deployment

**Trigger:** Every push to `main` branch

---

## 5. Model Deployment Strategy

**Strategy:** Real-time REST API (online serving)

**Justification Matrix:**

| Strategy | Latency | Throughput | Suitability for Fraud |
|----------|---------|------------|----------------------|
| Real-time API ✅ | < 100ms | High | ✅ Decision before transaction completes |
| Batch processing | Hours | Very high | ❌ Too slow — fraud already processed |
| Streaming | Low | Medium | Possible but complex for student account |

**Implementation:**
- FastAPI application (`app.py`) wraps the trained XGBoost model
- Accepts a single transaction JSON payload
- Returns: fraud probability, binary prediction, risk level (LOW/MEDIUM/HIGH)
- Deployed on Azure Container Instances (ACI) — serverless, no VM management needed

---

## 6. Cloud Deployment & Infrastructure

### Azure Services Used

| Service | Purpose |
|---------|---------|
| Azure Blob Storage | DVC data remote + model artifact store |
| Azure Container Registry (ACR) | Store versioned Docker images |
| Azure Container Instances (ACI) | Host REST API (serverless containers) |

### Infrastructure Architecture
```
Developer Machine
      │
      │ git push
      ▼
GitHub (source + CI/CD trigger)
      │
      │ GitHub Actions
      ▼
Azure Blob ◄── DVC pull ──► Training Pipeline
      │
      │ Docker image
      ▼
Azure ACR ──► Azure ACI (REST API endpoint)
                    │
                    ▼
              POST /predict
              (real-time inference)
```

### Cost Consideration (Student Account)
- Azure Blob Storage: ~$0.02/GB/month
- Azure ACI: ~$0.0025/vCPU-second (only charged when running)
- Azure ACR: Free tier for < 10GB storage
- **Total estimated cost: < $5/month for this project**

---

## 7. Monitoring, Logging & Governance

### Prediction Logging
Every inference call is logged to `logs/predictions.log`:
```
2024-01-16 10:23:45 [API] txn_id=txn_20240116_001 | prediction=1 |
probability=0.9823 | risk=HIGH | amount=149.62 | latency=12.4ms
```

### Runtime Metrics (GET /metrics endpoint)
- Total predictions served
- Fraud detection rate
- Average inference latency
- Error count

### Data Drift Detection (`src/pipeline/monitor.py`)
- Kolmogorov-Smirnov statistical test on each feature
- Compares live inference distribution against training baseline
- Flags features with statistically significant drift (p < 0.05)
- Output saved to `reports/drift_report.json`

### Governance (MLflow Model Registry)
- All models pass through: `None → Staging → Production`
- Models tagged with: AUC-ROC score, dataset version, approval status
- No model promoted to Production without human review
- Full audit trail of all model versions in MLflow UI

---

## 8. Results

| Metric | Value |
|--------|-------|
| AUC-ROC | ~0.98 |
| F1 Score (fraud) | ~0.87 |
| Precision | ~0.89 |
| Recall | ~0.85 |
| Avg API Latency | < 50ms |
| CI/CD Pipeline Time | ~8 minutes |

---

## 9. Reproducibility

Any reviewer can reproduce this project in full:

```bash
git clone https://github.com/YOUR_USERNAME/mlops-fraud-detection.git
cd mlops-fraud-detection
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add Azure credentials
dvc pull              # pull data from Azure Blob
dvc repro             # run full pipeline
mlflow ui             # view experiment tracking
uvicorn app:app       # run API locally
```

---

*Report generated as part of CA 4 Mini Project Assessment — MLOps Unit 2*
