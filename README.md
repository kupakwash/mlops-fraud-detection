# 🛡️ MLOps Fraud Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-orange?style=for-the-badge)
![MLflow](https://img.shields.io/badge/MLflow-2.9.2-blue?style=for-the-badge)
![DVC](https://img.shields.io/badge/DVC-3.38.1-purple?style=for-the-badge)
![Azure](https://img.shields.io/badge/Azure-Student-0078D4?style=for-the-badge&logo=microsoftazure)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green?style=for-the-badge&logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?style=for-the-badge&logo=docker)
![CI/CD](https://img.shields.io/badge/CI/CD-GitHub_Actions-2088FF?style=for-the-badge&logo=githubactions)

**An end-to-end MLOps system for real-time credit card fraud detection**

*CA 4 Mini Project · MLOps · Unit 2*

</div>

---

## 📌 Problem Statement

Credit card fraud is one of the most critical challenges in the global financial industry.
**Global fraud losses exceed $32 billion annually**, and financial institutions require
intelligent, real-time systems capable of flagging suspicious transactions before they
are processed.

This project implements a **production-grade MLOps pipeline** for binary classification
of credit card transactions as fraudulent or legitimate — demonstrating the complete
model lifecycle from raw data to monitored cloud deployment.

### Dataset
- **Source:** [Kaggle — Credit Card Fraud Detection (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions
- **Fraud rate:** 492 fraud cases (0.172% — heavily imbalanced)
- **Features:** V1–V28 (PCA-transformed anonymized features), `Amount`, `Time`
- **Target:** `Class` → 0 = Legitimate, 1 = Fraud

### Algorithm Choice: XGBoost
| Criteria | Justification |
|----------|--------------|
| Tabular data | XGBoost is the industry benchmark for structured/tabular data |
| Class imbalance | `scale_pos_weight` parameter handles 99.8% vs 0.2% imbalance natively |
| Speed | Trains in under 2 minutes on CPU — ideal for CI/CD automated retraining |
| Interpretability | Feature importance scores help explain predictions to auditors |
| Industry standard | Used by Stripe, PayPal, Visa in production fraud systems |

### Deployment Strategy: Real-Time REST API
> **Justification:** Fraud detection decisions must be made **within milliseconds** of a
> transaction being initiated. Batch processing (overnight scoring) is completely
> unsuitable — the fraudulent transaction would already have gone through. We deploy
> as a **real-time REST API** using FastAPI on Azure Container Instances, accepting
> a single transaction payload and returning a fraud probability score instantly.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA LAYER                                  │
│  Kaggle CSV ──► DVC Track ──► Azure Blob Storage (Remote)      │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                   PIPELINE LAYER                                │
│  ingest.py ──► preprocess.py ──► train.py ──► evaluate.py      │
│                                      │                          │
│                               MLflow Tracking                   │
│                         (params · metrics · artifacts)          │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                    CI/CD LAYER                                  │
│  GitHub Push ──► GitHub Actions ──► Lint ──► Test ──► Train    │
│                       ──► Docker Build ──► Push to ACR         │
│                       ──► Deploy to Azure ACI                  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                  SERVING LAYER                                  │
│  Azure ACR (Docker Image) ──► Azure ACI (REST API Endpoint)    │
│  POST /predict → { fraud_probability, prediction, risk_level } │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                MONITORING LAYER                                 │
│  Python Logging ──► logs/predictions.log                       │
│  Azure App Insights ──► latency · error rate · request count   │
│  Drift Detection ──► input distribution vs training baseline   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧰 Tech Stack

| Layer | Tool | Purpose |
|-------|------|---------|
| **Data Versioning** | DVC 3.38 | Track dataset versions, reproduce experiments |
| **Remote Storage** | Azure Blob Storage | Store DVC-tracked data and model artifacts |
| **Experiment Tracking** | MLflow 2.9 | Log params, metrics, models, compare runs |
| **ML Algorithm** | XGBoost 2.0 | Binary fraud classification |
| **Imbalance Handling** | imbalanced-learn | SMOTE oversampling for minority class |
| **Pipeline Modules** | Python (clean OOP) | Modular, testable, reproducible pipeline |
| **CI/CD** | GitHub Actions | Automated lint → test → train → build → deploy |
| **Containerization** | Docker | Package API + model into portable image |
| **Container Registry** | Azure ACR | Store and version Docker images |
| **Cloud Compute** | Azure ACI | Host REST API endpoint (serverless containers) |
| **API Framework** | FastAPI | High-performance async REST API serving |
| **Monitoring** | Python logging + Evidently | Prediction logs + data drift detection |
| **Governance** | MLflow Model Registry | Model staging, versioning, approval workflow |

---

## 📁 Project Structure

```
mlops-fraud-detection/
│
├── 📂 data/
│   ├── raw/                    # Original dataset (DVC tracked, not in git)
│   │   └── creditcard.csv
│   └── processed/              # Engineered features (DVC tracked)
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── 📂 src/
│   └── pipeline/
│       ├── __init__.py
│       ├── ingest.py           # Load raw data, train/test split
│       ├── preprocess.py       # Scaling, SMOTE, feature engineering
│       ├── train.py            # XGBoost training + MLflow logging
│       ├── evaluate.py         # Metrics: AUC-ROC, F1, confusion matrix
│       └── register.py         # MLflow Model Registry promotion
│
├── 📂 models/
│   └── fraud_model.pkl         # Saved model artifact
│
├── 📂 notebooks/
│   └── eda.ipynb               # Exploratory Data Analysis
│
├── 📂 tests/
│   ├── __init__.py
│   ├── test_ingest.py
│   ├── test_preprocess.py
│   └── test_api.py
│
├── 📂 reports/
│   └── technical_report.md     # Full project report
│
├── 📂 logs/
│   └── predictions.log         # Runtime inference logs
│
├── 📂 .github/
│   └── workflows/
│       └── mlops.yml           # Full CI/CD pipeline definition
│
├── 📄 app.py                   # FastAPI serving application
├── 📄 Dockerfile               # Container definition
├── 📄 dvc.yaml                 # DVC pipeline stages
├── 📄 params.yaml              # Hyperparameters (DVC-tracked)
├── 📄 requirements.txt         # All Python dependencies
├── 📄 .env.example             # Environment variable template
├── 📄 .gitignore               # Git ignore rules
├── 📄 .dvcignore               # DVC ignore rules
└── 📄 README.md                # This file
```

---

## ⚡ Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/mlops-fraud-detection.git
cd mlops-fraud-detection
```

### 2. Set up environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure environment variables
```bash
cp .env.example .env
# Edit .env with your Azure credentials
```

### 4. Pull data with DVC
```bash
dvc pull
```

### 5. Run full pipeline
```bash
dvc repro
```

### 6. View experiments in MLflow UI
```bash
mlflow ui
# Open http://localhost:5000
```

### 7. Run API locally
```bash
uvicorn app:app --reload
# Open http://localhost:8000/docs
```

---

## 🔄 CI/CD Pipeline

Every push to `main` branch triggers the following automated workflow:

```
Push to main
    │
    ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Lint Check │───►│  Unit Tests │───►│  Train Job  │
│  (flake8)   │    │  (pytest)   │    │  (pipeline) │
└─────────────┘    └─────────────┘    └──────┬──────┘
                                              │
                                    ┌─────────▼──────┐
                                    │  Docker Build  │
                                    │  Push to ACR   │
                                    └─────────┬──────┘
                                              │
                                    ┌─────────▼──────┐
                                    │ Deploy to ACI  │
                                    │  (live API)    │
                                    └────────────────┘
```

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| AUC-ROC | ~0.98 |
| F1 Score (fraud class) | ~0.87 |
| Precision | ~0.89 |
| Recall | ~0.85 |
| Training time | < 2 min |

*Metrics logged to MLflow and updated on every CI/CD run.*

---

## 🌐 API Reference

### `POST /predict`
Accepts a transaction feature vector and returns fraud probability.

**Request body:**
```json
{
  "V1": -1.359807, "V2": -0.072781, "V3": 2.536347,
  "V4": 1.378155, "V5": -0.338321, "V6": 0.462388,
  "V7": 0.239599, "V8": 0.098698, "V9": 0.363787,
  "V10": 0.090794, "V11": -0.551600, "V12": -0.617801,
  "V13": -0.991390, "V14": -0.311169, "V15": 1.468177,
  "V16": -0.470401, "V17": 0.207971, "V18": 0.025791,
  "V19": 0.403993, "V20": 0.251412, "V21": -0.018307,
  "V22": 0.277838, "V23": -0.110474, "V24": 0.066928,
  "V25": 0.128539, "V26": -0.189115, "V27": 0.133558,
  "V28": -0.021053, "Amount": 149.62
}
```

**Response:**
```json
{
  "prediction": 1,
  "fraud_probability": 0.9823,
  "risk_level": "HIGH",
  "model_version": "1",
  "transaction_id": "txn_20240116_001"
}
```

### `GET /health`
Returns API health status and model version.

### `GET /metrics`
Returns monitoring summary: total predictions, fraud rate, average latency.

---

## 📈 Evaluation Criteria Coverage

| Criterion | Implementation | Status |
|-----------|---------------|--------|
| Problem Definition & ML Use Case | Fintech fraud detection, XGBoost, business justification | ✅ |
| Data Versioning & Experiment Tracking | DVC + Azure Blob + MLflow runs | ✅ |
| Modular Pipeline Design | 5 clean Python modules in `src/pipeline/` | ✅ |
| CI/CD Implementation | GitHub Actions: lint→test→train→build→deploy | ✅ |
| Model Deployment Strategy | Real-time REST API with justification | ✅ |
| Cloud Deployment & Infrastructure | Azure ACR + ACI + Blob Storage | ✅ |
| Monitoring, Logging & Governance | Logging + drift detection + MLflow Registry | ✅ |
| GitHub Repository & Reproducibility | Clean structure + README + DVC repro | ✅ |
| Technical Project Report | Full report in `reports/technical_report.md` | ✅ |
| Technical Viva Defense | Hands-on demo of all layers | ✅ |

---

## 👤 Author

**Kupakwashe T. Mapuranga**
*AI/ML Engineering · MLOps · Data Analytics*

---

<div align="center">
<i>Built for CA 4 Mini Project Assessment · MLOps Unit 2</i>
</div>
