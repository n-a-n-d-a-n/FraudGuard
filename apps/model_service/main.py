import os
from contextlib import asynccontextmanager
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List

import model_loader
from explain import explain_transaction

# Global state for loaded model and metadata
model = None
model_name = ""
model_version = ""
model_source = ""


def unwrap_model(model_obj):
    """
    Extracts the underlying estimator if wrapped in CalibratedClassifierCV.
    """
    if hasattr(model_obj, "estimator") and model_obj.estimator is not None:
        return model_obj.estimator
    elif hasattr(model_obj, "base_estimator") and model_obj.base_estimator is not None:
        return model_obj.base_estimator
    elif hasattr(model_obj, "calibrated_classifiers_") and len(model_obj.calibrated_classifiers_) > 0:
        return model_obj.calibrated_classifiers_[0].estimator
    return model_obj


def determine_model_version(model_obj) -> str:
    """
    Derives model version tag based on loaded model type.
    """
    base = unwrap_model(model_obj)
    type_name = type(base).__name__
    if "XGB" in type_name:
        return "xgb-1.0"
    elif "LGBM" in type_name:
        return "lgbm-1.0"
    elif "LogisticRegression" in type_name:
        return "logreg-1.0"
    return f"{type_name.lower()}-1.0"


def load_model():
    """
    Loads saved model at startup using model_loader (MLflow Model Registry Staging stage with local joblib fallback).
    """
    global model, model_name, model_version, model_source
    model, model_source = model_loader.load_production_model()

    base = unwrap_model(model)
    model_name = type(base).__name__
    model_version = determine_model_version(base)
    print(f"[main] Loaded model '{model_name}' ({model_version}) via source '{model_source}'.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan event context manager to initialize model on service startup.
    """
    load_model()
    yield


app = FastAPI(
    title="FraudGuard 360 - Fraud Detection Model Service",
    description="ML Inference & SHAP Explanation Service for Member 4 module",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware enabling requests from all origins for demo/testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Any


# --- Pydantic Data Models ---
class TransactionFeatures(BaseModel):
    customer_txn_count_60m: int = Field(..., description="Number of transactions in last 1 hour (0-20)")
    amount_deviation_ratio: float = Field(..., description="Z-score / ratio deviation from normal spending (-1 to 5)")
    is_new_device: int = Field(..., description="Indicator if transaction uses a new device (0 or 1)")
    shared_device_account_count: int = Field(..., description="Count of other accounts sharing device (0-10)")

    @model_validator(mode="before")
    @classmethod
    def map_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            mapped = dict(data)
            if "velocity_1h" in mapped and "customer_txn_count_60m" not in mapped:
                mapped["customer_txn_count_60m"] = mapped["velocity_1h"]
            if "amount_deviation" in mapped and "amount_deviation_ratio" not in mapped:
                mapped["amount_deviation_ratio"] = mapped["amount_deviation"]
            if "new_device" in mapped and "is_new_device" not in mapped:
                mapped["is_new_device"] = mapped["new_device"]
            if "shared_device_count" in mapped and "shared_device_account_count" not in mapped:
                mapped["shared_device_account_count"] = mapped["shared_device_count"]
            return mapped
        return data

    @property
    def velocity_1h(self) -> int:
        return self.customer_txn_count_60m

    @property
    def amount_deviation(self) -> float:
        return self.amount_deviation_ratio

    @property
    def new_device(self) -> int:
        return self.is_new_device

    @property
    def shared_device_count(self) -> int:
        return self.shared_device_account_count


class PredictRequest(BaseModel):
    features: TransactionFeatures

    @model_validator(mode="before")
    @classmethod
    def extract_features(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "features" in data:
                return data
            elif "data" in data and isinstance(data["data"], dict):
                return {"features": data["data"]}
            else:
                # Direct flat feature payload
                return {"features": data}
        return data


class PredictResponse(BaseModel):
    fraud_probability: float
    risk_score: int
    prediction: str
    model_version: str


class ModelInfoResponse(BaseModel):
    model: str
    version: str
    source: str


class FactorImpact(BaseModel):
    feature: str
    impact: float


class ExplainResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    top_factors: List[FactorImpact]


# --- Endpoints ---

@app.get("/api/v1/health")
def health_check():
    """
    Health check endpoint returning service status.
    """
    return {"status": "ok"}


@app.post("/api/v1/model/predict", response_model=PredictResponse)
def predict_fraud(payload: PredictRequest):
    """
    Predicts transaction fraud probability, risk score, and risk category.
    Accepts Member 3 direct feature schema, Member 3 envelope payloads ('data'), and legacy field aliases.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    feats = payload.features
    df_single = pd.DataFrame([{
        "customer_txn_count_60m": feats.customer_txn_count_60m,
        "amount_deviation_ratio": feats.amount_deviation_ratio,
        "is_new_device": feats.is_new_device,
        "shared_device_account_count": feats.shared_device_account_count
    }])[["customer_txn_count_60m", "amount_deviation_ratio", "is_new_device", "shared_device_account_count"]]

    fraud_prob = float(model.predict_proba(df_single)[0, 1])
    prob_rounded = round(fraud_prob, 4)
    risk_score = int(round(fraud_prob * 100))

    if fraud_prob >= 0.7:
        prediction = "HIGH_RISK"
    elif fraud_prob >= 0.4:
        prediction = "MEDIUM_RISK"
    else:
        prediction = "LOW_RISK"

    return PredictResponse(
        fraud_probability=prob_rounded,
        risk_score=risk_score,
        prediction=prediction,
        model_version=model_version
    )


import urllib.request
import json

MEMBER3_FEATURE_SERVICE_URL = os.getenv("MEMBER3_FEATURE_SERVICE_URL", "http://localhost:8003/api/v1/features/extract")


class FeatureServicePredictRequest(BaseModel):
    transaction: dict
    feature_service_url: Optional[str] = None


@app.get("/api/v1/model/info", response_model=ModelInfoResponse)
def get_model_info():
    """
    Returns information about the loaded model type, version tag, and model source ('registry' or 'local_fallback').
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    return ModelInfoResponse(
        model=model_name,
        version=model_version,
        source=model_source
    )


@app.post("/api/v1/model/predict_via_feature_service", response_model=PredictResponse)
def predict_via_feature_service(payload: FeatureServicePredictRequest):
    """
    Calls Member 3 Feature Extraction Service at http://localhost:8003/api/v1/features/extract
    to extract features for a raw transaction, then performs real-time ML fraud prediction.
    """
    service_url = payload.feature_service_url or MEMBER3_FEATURE_SERVICE_URL
    
    try:
        req_data = json.dumps(payload.transaction).encode("utf-8")
        req = urllib.request.Request(
            service_url,
            data=req_data,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            m3_response = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to communicate with Member 3 Feature Service at '{service_url}': {str(e)}"
        )

    # Unwrap Member 3 response envelope and pass to predict_fraud logic
    predict_req = PredictRequest.model_validate(m3_response)
    return predict_fraud(predict_req)


@app.get("/api/v1/model/explain/{transaction_id}", response_model=ExplainResponse)
def explain_tx(
    transaction_id: str,
    customer_txn_count_60m: Optional[int] = Query(None, description="Transactions in last hour (Member 3)"),
    amount_deviation_ratio: Optional[float] = Query(None, description="Amount deviation ratio (Member 3)"),
    is_new_device: Optional[int] = Query(None, description="New device indicator (Member 3)"),
    shared_device_account_count: Optional[int] = Query(None, description="Shared device count (Member 3)"),
    velocity_1h: Optional[int] = Query(None, description="Legacy velocity_1h"),
    amount_deviation: Optional[float] = Query(None, description="Legacy amount_deviation"),
    new_device: Optional[int] = Query(None, description="Legacy new_device"),
    shared_device_count: Optional[int] = Query(None, description="Legacy shared_device_count")
):
    """
    Computes SHAP feature attribution and explanations for a transaction specified by features.
    """
    vel = customer_txn_count_60m if customer_txn_count_60m is not None else velocity_1h
    amt = amount_deviation_ratio if amount_deviation_ratio is not None else amount_deviation
    dev = is_new_device if is_new_device is not None else new_device
    shd = shared_device_account_count if shared_device_account_count is not None else shared_device_account_count

    if vel is None or amt is None or dev is None or shd is None:
        raise HTTPException(status_code=422, detail="Missing required transaction features for explanation.")

    feature_dict = {
        "customer_txn_count_60m": vel,
        "amount_deviation_ratio": amt,
        "is_new_device": dev,
        "shared_device_account_count": shd
    }

    try:
        explanation = explain_transaction(feature_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate SHAP explanation: {str(e)}")

    return ExplainResponse(
        transaction_id=transaction_id,
        fraud_probability=explanation["fraud_probability"],
        top_factors=[
            FactorImpact(feature=item["feature"], impact=item["impact"])
            for item in explanation["top_factors"]
        ]
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
