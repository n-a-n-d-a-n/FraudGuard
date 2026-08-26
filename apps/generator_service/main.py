from fastapi import FastAPI
from uuid import uuid4
from datetime import datetime, timezone
import pandas as pd
from shared.schemas.envelope import EnvelopeResponse, EnvelopeRequest
from shared.schemas.transaction import GenerationRequest, DatasetMetadata
from apps.generator_service.generators.baseline import generate_baseline_transactions
from apps.generator_service.generators.attack_conditioner import generate_attack_transactions
from apps.generator_service.storage import save_dataset
from apps.generator_service.validators.data_validator import validate_dataset
from fastapi import HTTPException
import os
import json     
from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError
from fastapi import Request
from fastapi.responses import JSONResponse
from shared.schemas.envelope import StatusEnum, ErrorDetail
from apps.generator_service.generators.ctgan import train_ctgan_model, generate_ctgan_rows

app = FastAPI(title="FraudGuard 360 - Synthetic Generator", docs_url="/docs")

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=422,
        content=EnvelopeResponse(
            request_id="unknown",
            timestamp=datetime.now(timezone.utc),
            status=StatusEnum.ERROR,
            error=ErrorDetail(code="VALIDATION_ERROR", message=str(exc))
        ).model_dump(mode="json")
    )

@app.exception_handler(IntegrityError)
async def integrity_exception_handler(request: Request, exc: IntegrityError):
    return JSONResponse(
        status_code=409,
        content=EnvelopeResponse(
            request_id="unknown",
            timestamp=datetime.now(timezone.utc),
            status=StatusEnum.ERROR,
            error=ErrorDetail(code="INTEGRITY_ERROR", message="Database constraint violated (e.g., duplicate ID).")
        ).model_dump(mode="json")
    )

def generate_request_id() -> str:
    return f"REQ_{uuid4().hex[:12]}"

@app.get("/health")
async def health():
    return EnvelopeResponse(
        request_id=generate_request_id(),
        timestamp=datetime.now(timezone.utc),
        data={"status": "healthy"}
    )



@app.post("/api/v1/generator/transactions")
async def generate_transactions(payload: EnvelopeRequest):
    req_data = GenerationRequest(**payload.data)
    
    # Calculate row counts
    fraud_rows_count = int(req_data.rows * req_data.fraud_ratio)
    legit_rows_count = req_data.rows - fraud_rows_count
    
    # 1. Generate Legitimate Transactions
    if req_data.generator_type == "ctgan":
        df_legit = generate_ctgan_rows(rows=legit_rows_count, seed=req_data.seed)
    else:
        df_legit = generate_baseline_transactions(rows=legit_rows_count, seed=req_data.seed)
    
    # CRITICAL FIX: Extract the customer pool from the legit data!
    existing_customers = df_legit["customer_id"].unique().tolist()
    
    # 2. Generate Fraud Transactions (if requested)
    df_fraud = pd.DataFrame()
    if fraud_rows_count > 0 and req_data.attack_ids:
        fraud_dfs = []
        rows_per_attack = fraud_rows_count // len(req_data.attack_ids)
        
        for i, attack_id in enumerate(req_data.attack_ids):
            rows = rows_per_attack + (fraud_rows_count % len(req_data.attack_ids) if i == len(req_data.attack_ids)-1 else 0)
            if rows > 0:
                # Pass the customer pool to the attack generator!
                fraud_dfs.append(generate_attack_transactions(attack_id, rows=rows, seed=req_data.seed + i + 1, existing_customers=existing_customers))
                
        if fraud_dfs:
            df_fraud = pd.concat(fraud_dfs).sample(frac=1, random_state=req_data.seed).reset_index(drop=True)

    # 3. Combine and shuffle
    df_combined = pd.concat([df_legit, df_fraud]).sample(frac=1, random_state=req_data.seed).reset_index(drop=True)
    
    dataset_id = f"DS_{uuid4().hex[:8]}"
    
    # Create the metadata object first
    metadata = DatasetMetadata(
        dataset_id=dataset_id,
        rows=len(df_combined),
        fraud_rows=len(df_fraud),
        schema_version="1.0",
        attack_ids=req_data.attack_ids if len(df_fraud) > 0 else [],
        seed=req_data.seed,
        generator_version="1.0.0",
        provenance="baseline_plus_attacks",
        created_at=datetime.now(timezone.utc)
    )
    
    # 4. Run Validators
    validation_report = validate_dataset(df_combined, metadata)
    
    # NEW: Add Advanced Fidelity Metrics
    from apps.generator_service.validators.data_validator import calculate_fidelity_metrics
    validation_report["fidelity_metrics"] = calculate_fidelity_metrics(df_combined)
    
    if not validation_report["schema_valid"]:
        raise HTTPException(status_code=500, detail=f"Data validation failed: {validation_report['quality_issues']}")
    
    # 5. Save to Parquet and Manifest
    save_dataset(df_combined, metadata)
    
    # 6. Return response with validation report included
    response_data = metadata.model_dump()
    response_data["validation_report"] = validation_report
    
    return EnvelopeResponse(
        request_id=payload.request_id,
        timestamp=datetime.now(timezone.utc),
        data=response_data
    )

@app.get("/api/v1/generator/dataset/{dataset_id}")
async def get_dataset(dataset_id: str):
    # Look for the manifest file we saved earlier
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    manifest_path = os.path.join(project_root, "data", "synthetic", dataset_id, "manifest.json")
    
    if not os.path.exists(manifest_path):
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    with open(manifest_path, "r") as f:
        metadata = json.load(f)
        
    return EnvelopeResponse(
        request_id=generate_request_id(),
        timestamp=datetime.now(timezone.utc),
        data=metadata
    )

@app.post("/api/v1/generator/scenario")
async def generate_scenario_dataset(payload: EnvelopeRequest):
    """Generates a dataset for a single specific scenario."""
    req_data = GenerationRequest(**payload.data)
    
    if not req_data.attack_ids:
        raise HTTPException(status_code=400, detail="attack_ids is required for scenario generation")
        
    attack_id = req_data.attack_ids[0]
    df_fraud = generate_attack_transactions(attack_id, rows=req_data.rows, seed=req_data.seed)
    
    dataset_id = f"DS_{uuid4().hex[:8]}"
    metadata = DatasetMetadata(
        dataset_id=dataset_id,
        rows=len(df_fraud),
        fraud_rows=len(df_fraud),
        schema_version="1.0",
        attack_ids=[attack_id],
        seed=req_data.seed,
        generator_version="1.0.0",
        provenance=f"scenario_only_{attack_id}",
        created_at=datetime.now(timezone.utc)
    )
    
    validation_report = validate_dataset(df_fraud, metadata)
    if not validation_report["schema_valid"]:
        raise HTTPException(status_code=500, detail=f"Data validation failed: {validation_report['quality_issues']}")
        
    save_dataset(df_fraud, metadata)
    
    response_data = metadata.model_dump()
    response_data["validation_report"] = validation_report
    
    return EnvelopeResponse(
        request_id=payload.request_id,
        timestamp=datetime.now(timezone.utc),
        data=response_data
    )

@app.post("/api/v1/generator/train_ctgan")
async def train_ctgan():
    """Triggers CTGAN training in the background."""
    train_ctgan_model()
    return EnvelopeResponse(
        request_id=generate_request_id(),
        timestamp=datetime.now(timezone.utc),
        data={"status": "success", "message": "CTGAN model trained and saved."}
    )