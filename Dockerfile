# ── Base image ────────────────────────────────────────────────────
FROM python:3.11-slim

# ── Metadata ──────────────────────────────────────────────────────
LABEL maintainer="Kupakwashe T. Mapuranga"
LABEL description="Credit Card Fraud Detection API - MLOps CA4 Project"
LABEL version="1.0.0"

# ── System deps ───────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────
WORKDIR /app

# ── Copy requirements first (Docker layer caching) ────────────────
COPY requirements.txt .

# ── Install Python dependencies ───────────────────────────────────
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        pandas==2.2.3 \
        numpy==2.4.4 \
        scikit-learn==1.8.0 \
        xgboost==3.2.0 \
        joblib==1.5.3 \
        fastapi==0.135.3 \
        uvicorn==0.44.0 \
        pydantic==2.13.1 \
        pyyaml==6.0.3 \
        python-dotenv==1.2.2

# ── Copy application files ────────────────────────────────────────
COPY app.py .
COPY params.yaml .
COPY models/ ./models/

# ── Create logs directory ─────────────────────────────────────────
RUN mkdir -p logs

# ── Environment variables ─────────────────────────────────────────
ENV MODEL_PATH=models/fraud_model.pkl
ENV SCALER_PATH=models/scaler.pkl
ENV MODEL_VERSION=1
ENV PYTHONUNBUFFERED=1

# ── Expose port ───────────────────────────────────────────────────
EXPOSE 8000

# ── Health check ──────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# ── Start command ─────────────────────────────────────────────────
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
