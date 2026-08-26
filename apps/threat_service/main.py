from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from uuid import uuid4
from datetime import datetime, timezone
from shared.schemas.envelope import EnvelopeResponse, EnvelopeRequest, ErrorDetail, StatusEnum
from shared.schemas.attack import AttackScenario, MutationRequest, MutationResponse
from apps.threat_service.database import engine, Base, get_db
from apps.threat_service.models import AttackScenarioDB
from dotenv import load_dotenv
load_dotenv()
from apps.threat_service.mutation.operators import apply_mutations
from shared.schemas.attack import AttackScenario, Provenance
from shared.schemas.attack import AttackScenario, Provenance, SIGNAL_MAP
from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError
from fastapi import Request
from fastapi.responses import JSONResponse

app = FastAPI(title="FraudGuard 360 - Threat Intelligence", docs_url="/docs")

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

# Create tables on startup
@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)

def generate_request_id() -> str:
    return f"REQ_{uuid4().hex[:12]}"

@app.get("/health")
async def health():
    return EnvelopeResponse(
        request_id=generate_request_id(),
        timestamp=datetime.now(timezone.utc),
        data={"status": "healthy"}
    )

@app.post("/api/v1/attacks/discover")
async def discover_attacks(payload: EnvelopeRequest, db: Session = Depends(get_db)):
    from apps.threat_service.llm.discover_agent import discover_chain
    
    target_category = payload.data.get("category", "ACCOUNT_TAKEOVER")
    result = discover_chain.invoke({"category": target_category})
    
    if result.get("error"):
        return EnvelopeResponse(
            request_id=payload.request_id,
            timestamp=datetime.now(timezone.utc),
            data={"status": "failed", "error": result["error"], "candidate": None}
        )
        
    candidate = result.get("validated_scenario")
    
    # NEW: Save the discovered scenario to the database!
    if candidate:
        db_scenario = AttackScenarioDB(**candidate)
        db.add(db_scenario)
        db.commit()
        db.refresh(db_scenario)
        candidate = AttackScenario.model_validate(db_scenario).model_dump()
    
    return EnvelopeResponse(
        request_id=payload.request_id,
        timestamp=datetime.now(timezone.utc),
        data={"status": "success", "candidate": candidate}
    )

@app.get("/api/v1/attacks")
async def list_attacks(db: Session = Depends(get_db)):
    attacks = db.query(AttackScenarioDB).all()
    # Convert SQLAlchemy objects to Pydantic schemas
    attacks_data = [AttackScenario.model_validate(a) for a in attacks]
    return EnvelopeResponse(
        request_id=generate_request_id(),
        timestamp=datetime.now(timezone.utc),
        data={"count": len(attacks_data), "attacks": attacks_data}
    )


@app.get("/api/v1/attacks/{attack_id}")
async def get_attack(attack_id: str, db: Session = Depends(get_db)):
    attack = db.query(AttackScenarioDB).filter(AttackScenarioDB.attack_id == attack_id).first()
    if not attack:
        raise HTTPException(status_code=404, detail="Attack not found")
    # Convert SQLAlchemy object to Pydantic schema
    attack_data = AttackScenario.model_validate(attack)
    return EnvelopeResponse(
        request_id=generate_request_id(),
        timestamp=datetime.now(timezone.utc),
        data=attack_data
    )


@app.post("/api/v1/attacks/mutate")
async def mutate_attack(payload: EnvelopeRequest, db: Session = Depends(get_db)):
    req_data = MutationRequest(**payload.data)
    
    # 1. Fetch the parent scenario
    parent = db.query(AttackScenarioDB).filter(AttackScenarioDB.attack_id == req_data.attack_id).first()
    if not parent:
        raise HTTPException(status_code=404, detail="Parent attack not found")
        
    parent_schema = AttackScenario.model_validate(parent)
    
    # 2. Apply mutations to parameters
    new_params, applied_ops = apply_mutations(parent_schema.parameters, req_data.operators)
    
    # 3. Generate new ID and Version
    # e.g., ATO_001 -> ATO_001_V2
    base_id = parent_schema.attack_id.split("_V")[0]
    existing_variants = db.query(AttackScenarioDB).filter(AttackScenarioDB.attack_id.like(f"{base_id}_V%")).count()
    new_version_num = existing_variants + 1
    
    new_attack_id = f"{base_id}_V{new_version_num}"
    new_version_str = f"1.{new_version_num}"
    
    # 4. Create the new mutated scenario object
    mutated_scenario = AttackScenario(
        attack_id=new_attack_id,
        version=new_version_str,
        category=parent_schema.category,
        channel=parent_schema.channel,
        risk_level=parent_schema.risk_level,
        description=f"Mutated variant of {parent_schema.attack_id}. Harder pattern: {', '.join(applied_ops)}",
        parameters=new_params,
        features=[SIGNAL_MAP[k] for k in new_params.model_dump() if k in SIGNAL_MAP and getattr(new_params, k)],
        novelty_score=max(0.0, parent_schema.novelty_score - 0.1), # slightly less novel than baseline
        provenance=Provenance(
            source="mutation_engine",
            parent_attack_id=parent_schema.attack_id,
            mutation_operators=applied_ops
        )
    )
    
    # 5. Save to Database
    db_scenario = AttackScenarioDB(**mutated_scenario.model_dump())
    db.add(db_scenario)
    db.commit()
    db.refresh(db_scenario)
    
    return EnvelopeResponse(
        request_id=payload.request_id,
        timestamp=datetime.now(timezone.utc),
        data=AttackScenario.model_validate(db_scenario)
    )