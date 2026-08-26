from sqlalchemy import Column, String, Float, Boolean, DateTime, JSON
from datetime import datetime, timezone
from apps.threat_service.database import Base

class AttackScenarioDB(Base):
    __tablename__ = "attack_scenarios"

    attack_id = Column(String, primary_key=True, index=True)
    version = Column(String, primary_key=True, index=True)
    category = Column(String, index=True)
    channel = Column(String)
    risk_level = Column(String)
    description = Column(String)
    parameters = Column(JSON)
    features = Column(JSON)
    novelty_score = Column(Float, default=0.0)
    provenance = Column(JSON)
    simulation_ready = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))