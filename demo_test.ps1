# ── Demo Test Transactions ────────────────────────────────────────
# Three transactions to demonstrate the live API during viva
# Run each one and show the different fraud_probability scores

# ── SETUP: set your API IP here ───────────────────────────────────
$API_IP = "fraud-detection-kupak6.southeastasia.azurecontainer.io"
$BASE_URL = "http://$API_IP"

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  FRAUD DETECTION API - LIVE DEMO" -ForegroundColor Cyan
Write-Host "  $BASE_URL" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# ── HEALTH CHECK ─────────────────────────────────────────────────
Write-Host ""
Write-Host "[ HEALTH CHECK ]" -ForegroundColor Yellow
$health = Invoke-RestMethod -Uri "$BASE_URL/health" -Method GET
Write-Host "Status       : $($health.status)" -ForegroundColor Green
Write-Host "Model Loaded : $($health.model_loaded)" -ForegroundColor Green
Write-Host "Model Version: $($health.model_version)" -ForegroundColor Green

# ── TRANSACTION 1: HIGH FRAUD RISK ───────────────────────────────
Write-Host ""
Write-Host "[ TEST 1: Suspected Fraud Transaction ]" -ForegroundColor Red
Write-Host "Amount: $2.69 | Unusual V feature pattern"
$t1 = '{"Time":406.0,"V1":-2.3122,"V2":1.9519,"V3":-1.6096,"V4":3.9979,"V5":-0.5222,"V6":-1.4265,"V7":-2.5374,"V8":-0.2569,"V9":-0.1684,"V10":-1.4919,"V11":2.0798,"V12":-1.0543,"V13":-0.4627,"V14":-2.5615,"V15":1.3412,"V16":-1.1196,"V17":-1.0594,"V18":-0.6840,"V19":1.9658,"V20":-1.1563,"V21":0.4243,"V22":-0.2945,"V23":0.0277,"V24":0.5006,"V25":-0.4387,"V26":-0.1353,"V27":-0.5550,"V28":-0.0702,"Amount":2.69,"Time":406.0}'
$r1 = Invoke-RestMethod -Uri "$BASE_URL/predict" -Method POST -ContentType "application/json" -Body $t1
Write-Host "Prediction       : $($r1.prediction) (1=Fraud, 0=Legit)" -ForegroundColor Red
Write-Host "Fraud Probability: $($r1.fraud_probability)" -ForegroundColor Red
Write-Host "Risk Level       : $($r1.risk_level)" -ForegroundColor Red
Write-Host "Transaction ID   : $($r1.transaction_id)" -ForegroundColor Red
Write-Host "Latency          : $($r1.latency_ms)ms" -ForegroundColor Red

# ── TRANSACTION 2: LOW FRAUD RISK ────────────────────────────────
Write-Host ""
Write-Host "[ TEST 2: Legitimate Transaction ]" -ForegroundColor Green
Write-Host "Amount: $149.62 | Normal shopping pattern"
$t2 = '{"Time":0.0,"V1":-0.1221,"V2":0.3853,"V3":0.4407,"V4":0.4098,"V5":-0.0410,"V6":0.2107,"V7":0.0257,"V8":0.4033,"V9":0.3222,"V10":-0.0023,"V11":0.4145,"V12":-0.0698,"V13":-0.0794,"V14":0.0545,"V15":0.0555,"V16":0.0257,"V17":-0.0348,"V18":0.1500,"V19":0.0442,"V20":0.0175,"V21":0.0175,"V22":-0.0436,"V23":0.0045,"V24":0.0193,"V25":0.1499,"V26":-0.0688,"V27":0.0058,"V28":0.0003,"Amount":149.62}'
$r2 = Invoke-RestMethod -Uri "$BASE_URL/predict" -Method POST -ContentType "application/json" -Body $t2
Write-Host "Prediction       : $($r2.prediction) (1=Fraud, 0=Legit)" -ForegroundColor Green
Write-Host "Fraud Probability: $($r2.fraud_probability)" -ForegroundColor Green
Write-Host "Risk Level       : $($r2.risk_level)" -ForegroundColor Green
Write-Host "Transaction ID   : $($r2.transaction_id)" -ForegroundColor Green
Write-Host "Latency          : $($r2.latency_ms)ms" -ForegroundColor Green

# ── TRANSACTION 3: MEDIUM RISK ───────────────────────────────────
Write-Host ""
Write-Host "[ TEST 3: Borderline / Medium Risk Transaction ]" -ForegroundColor Yellow
Write-Host "Amount: $1,200.00 | Large unusual purchase"
$t3 = '{"Time":75432.0,"V1":-1.1521,"V2":0.8774,"V3":0.2243,"V4":1.1030,"V5":-0.3124,"V6":-0.4567,"V7":-0.6723,"V8":0.1234,"V9":-0.4321,"V10":-0.5678,"V11":0.7654,"V12":-0.3456,"V13":0.1234,"V14":-0.9876,"V15":0.4567,"V16":-0.3210,"V17":-0.2109,"V18":-0.1098,"V19":0.4567,"V20":-0.3456,"V21":0.1234,"V22":-0.0987,"V23":0.0123,"V24":0.1567,"V25":-0.1234,"V26":-0.0456,"V27":-0.2109,"V28":-0.0234,"Amount":1200.00}'
$r3 = Invoke-RestMethod -Uri "$BASE_URL/predict" -Method POST -ContentType "application/json" -Body $t3
Write-Host "Prediction       : $($r3.prediction) (1=Fraud, 0=Legit)" -ForegroundColor Yellow
Write-Host "Fraud Probability: $($r3.fraud_probability)" -ForegroundColor Yellow
Write-Host "Risk Level       : $($r3.risk_level)" -ForegroundColor Yellow
Write-Host "Transaction ID   : $($r3.transaction_id)" -ForegroundColor Yellow
Write-Host "Latency          : $($r3.latency_ms)ms" -ForegroundColor Yellow

# ── MONITORING METRICS ────────────────────────────────────────────
Write-Host ""
Write-Host "[ LIVE MONITORING METRICS ]" -ForegroundColor Cyan
$metrics = Invoke-RestMethod -Uri "$BASE_URL/metrics" -Method GET
Write-Host "Total Predictions: $($metrics.total_predictions)" -ForegroundColor Cyan
Write-Host "Fraud Detected   : $($metrics.fraud_predictions)" -ForegroundColor Cyan
Write-Host "Fraud Rate       : $($metrics.fraud_rate)" -ForegroundColor Cyan
Write-Host "Avg Latency (ms) : $($metrics.average_latency_ms)" -ForegroundColor Cyan

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  DEMO COMPLETE" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
