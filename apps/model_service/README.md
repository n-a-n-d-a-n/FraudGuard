# FraudGuard 360 - Fraud Detection ML Module (Member 4)

## 1. Overview
The Fraud Detection ML module (Member 4) for **FraudGuard 360** trains machine learning classification models on transaction features, calibrates prediction probabilities, registers production models to the MLflow Model Registry, and serves real-time fraud predictions, risk scoring, risk categorization, and SHAP-based feature explanations via a RESTful FastAPI service.

---

## 2. Setup Instructions

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Install project dependencies
pip install -r requirements.txt
```

---

## 3. How to Run

```powershell
# 1. Generate synthetic dummy training dataset
python dummy_data.py

# 2. Train candidate models, apply probability calibration, and register best model to MLflow Registry
python train.py

# 3. Run unit and API contract test suite
pytest tests/test_model.py -v

# 4. Start production FastAPI inference server (runs on http://localhost:8000)
python main.py

# 5. Launch MLflow UI to inspect experiment runs & Model Registry (opens http://127.0.0.1:5000)
mlflow ui
```
> Interactive API Documentation (Swagger UI) is available at: `http://localhost:8000/docs`

---

## 4. API Documentation

| Method | Endpoint Path | Description / Purpose |
| :--- | :--- | :--- |
| `POST` | `/api/v1/model/predict` | Predicts fraud probability, 0-100 risk score, risk level, and model version tag. |
| `GET` | `/api/v1/model/info` | Returns active model type, version, and `source` (`registry` or `local_fallback`). |
| `GET` | `/api/v1/model/explain/{transaction_id}` | Computes SHAP feature contributions for transaction query parameters. |
| `GET` | `/api/v1/health` | Service health check endpoint returning `{"status": "ok"}`. |

---

## 5. Sample Input / Output

### 5.1 Endpoint: `POST /api/v1/model/predict`

**Request Body (`application/json` - Member 3 Direct Payload):**
```json
{
  "features": {
    "customer_txn_count_60m": 8,
    "amount_deviation_ratio": 3.0,
    "is_new_device": 1,
    "shared_device_account_count": 4
  }
}
```

*Note: Also accepts Member 3 envelope format (`{"data": {...}}`) and flat JSON directly.*

**Response (`200 OK`):**
```json
{
  "fraud_probability": 1.0,
  "risk_score": 100,
  "prediction": "HIGH_RISK",
  "model_version": "logreg-1.0"
}
```

### 5.2 Endpoint: `GET /api/v1/model/info`

**Response (`200 OK`):**
```json
{
  "model": "LogisticRegression",
  "version": "logreg-1.0",
  "source": "registry"
}
```

---

## 6. Model Details
- **Benchmarked Classifiers**:
  - Baseline: `LogisticRegression(max_iter=1000)`
  - XGBoost: `XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1)`
  - LightGBM: `LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.1)`
- **Probability Calibration**: Top candidate selected by ROC AUC is wrapped with `CalibratedClassifierCV(estimator=best_model, method="sigmoid", cv=5)` to output well-calibrated posterior probabilities (Platt Scaling).
- **MLflow Model Registry**: Calibrated model registered under `"fraudguard360-detector"` and transitioned to stage `"Staging"` using `MlflowClient`.
- **Production Model Sourcing**: [model_loader.py](model_loader.py) loads from the MLflow Model Registry (`models:/fraudguard360-detector/Staging`) first, gracefully falling back to [models/best_model.joblib](models/best_model.joblib) if the registry is unreachable.

---

## 7. Known Limitations
1. **Provisional Decision Thresholds**: Risk category thresholds (`HIGH_RISK` >= 0.7, `MEDIUM_RISK` >= 0.4, `LOW_RISK` < 0.4) are provisional defaults pending joint calibration with Member 5.
2. **SHAP Feature Attribution Scale**: SHAP impacts for linear model baselines are evaluated on log-odds scale relative to reference background data.
3. **Staging Registry Stage**: Model registry stage is initialized to `"Staging"` - promotion to `"Production"` is a manual/deliberate step, not automated.
4. **Local Tracking Store**: MLflow uses a local `sqlite:///mlflow.db` backend store for this prototype rather than a shared remote MLflow server.

---

## 8. Dependencies & Team Handoff
- **Upstream Dependency**: Member 3 (`fraudguard-member3` feature engineering service: supplies `customer_txn_count_60m`, `amount_deviation_ratio`, `is_new_device`, `shared_device_account_count`).
- **Downstream Handoffs**:
  - **Member 5** (Risk API Integration & Gateway Orchestration consuming `/api/v1/model/predict` and `/api/v1/model/explain/{transaction_id}`, joint threshold tuning).
  - **Member 6** (Metrics & Operational Monitoring consuming `/api/v1/health` and `/api/v1/model/info`, retraining feedback loop).

---

## 9. Real Data Validation Findings
The Fraud Detection ML module was evaluated against the real `DS_7b49892c` dataset using `real_data_pipeline.py`. Validation revealed that due to low repeat-customer and repeat-device density in that dataset (only 51 of 946 customers and 37 of 960 devices have repeat transactions), 12 of 12 behavioral and graph features showed no meaningful separation between fraudulent and legitimate classes. This represents an upstream data generation issue (Member 2) rather than a modeling deficiency. Consequently, the module continues to utilize synthetic dataset generation (`dummy_data.py`) for its active demo model until updated transaction fixtures are available.


