from apps.threat_service.database import SessionLocal
from apps.threat_service.models import AttackScenarioDB
from shared.schemas.attack import AttackScenario, AttackCategory, RiskLevel
from shared.schemas.attack import AttackScenario, AttackCategory, RiskLevel, SIGNAL_MAP

# 12 Hand-written baseline scenarios
scenarios = [
    {"attack_id": "ATO_001", "category": AttackCategory.ACCOUNT_TAKEOVER, "channel": "CARD", "risk_level": RiskLevel.HIGH, "description": "Novel device + credential reset + abnormal velocity + new beneficiary", "parameters": {"new_device": True, "velocity_multiplier": 4, "merchant_novelty": True}},
    {"attack_id": "SYNID_001", "category": AttackCategory.SYNTHETIC_IDENTITY, "channel": "CARD", "risk_level": RiskLevel.HIGH, "description": "Inconsistent identity attributes + low-history account + unusual lifecycle", "parameters": {"amount_anomaly_z": 2.0, "merchant_novelty": True}},
    {"attack_id": "CNP_001", "category": AttackCategory.CARD_NOT_PRESENT, "channel": "CARD", "risk_level": RiskLevel.CRITICAL, "description": "New device/location + merchant mismatch + velocity anomalies", "parameters": {"new_device": True, "location_shift": True, "velocity_multiplier": 3}},
    {"attack_id": "UPI_001", "category": AttackCategory.UPI_SCAM, "channel": "UPI", "risk_level": RiskLevel.HIGH, "description": "New payee + unusual amount/time + repeated failed attempts", "parameters": {"velocity_multiplier": 5, "time_window_minutes": 15}},
    {"attack_id": "MULE_001", "category": AttackCategory.MULE_NETWORK, "channel": "P2P", "risk_level": RiskLevel.CRITICAL, "description": "Multiple senders to hub account + fast onward movement", "parameters": {"velocity_multiplier": 6, "time_window_minutes": 30, "shared_device": True}},
    {"attack_id": "BOT_001", "category": AttackCategory.BOT_ACCOUNT_TESTING, "channel": "CARD", "risk_level": RiskLevel.MEDIUM, "description": "High request velocity + repeated small attempts + device fingerprints", "parameters": {"velocity_multiplier": 10, "amount_anomaly_z": -1.5, "time_window_minutes": 5}},
    {"attack_id": "REFUND_001", "category": AttackCategory.REFUND_ABUSE, "channel": "CARD", "risk_level": RiskLevel.MEDIUM, "description": "Unusual refund ratio + transaction clustering + device overlap", "parameters": {"velocity_multiplier": 2, "merchant_novelty": False}},
    {"attack_id": "FF_001", "category": AttackCategory.FRIENDLY_FRAUD, "channel": "CARD", "risk_level": RiskLevel.LOW, "description": "Dispute behavior inconsistent with prior purchase/device history", "parameters": {"amount_anomaly_z": 0.5, "merchant_novelty": False}},
    {"attack_id": "LAUNDER_001", "category": AttackCategory.TRANSACTION_LAUNDERING, "channel": "CARD", "risk_level": RiskLevel.HIGH, "description": "Short holding time + split/merge patterns + rapid graph expansion", "parameters": {"velocity_multiplier": 4, "time_window_minutes": 45}},
    {"attack_id": "PROMO_001", "category": AttackCategory.PROMOTION_ABUSE, "channel": "WALLET", "risk_level": RiskLevel.MEDIUM, "description": "Many accounts/devices sharing identity or behavioral features", "parameters": {"new_device": True, "velocity_multiplier": 3}},
    {"attack_id": "KYC_001", "category": AttackCategory.KYC_SIMULATION, "channel": "UPI", "risk_level": RiskLevel.HIGH, "description": "Synthetic identity artifacts and inconsistent cross-field patterns", "parameters": {"location_shift": True, "merchant_novelty": True}},
    {"attack_id": "IMPERS_001", "category": AttackCategory.IMPERSONATION_SOCIAL_ENGINEERING, "channel": "P2P", "risk_level": RiskLevel.CRITICAL, "description": "Synthetic message context -> abnormal payee change -> urgent payment", "parameters": {"velocity_multiplier": 2, "amount_anomaly_z": 3.0, "time_window_minutes": 10}}
]


def seed_database():
    db = SessionLocal()
    try:
        # Clear existing data
        db.query(AttackScenarioDB).delete()
        
        for scenario_data in scenarios:
            # Dynamically generate features list based on which parameters are actually present
            active_features = [SIGNAL_MAP[k] for k in scenario_data["parameters"] if k in SIGNAL_MAP]
            
            # Use Pydantic to validate, then dump to dict for DB
            scenario = AttackScenario(
                attack_id=scenario_data["attack_id"],
                category=scenario_data["category"],
                channel=scenario_data["channel"],
                risk_level=scenario_data["risk_level"],
                description=scenario_data["description"],
                parameters=scenario_data["parameters"],
                features=active_features,
                novelty_score=0.9, # High novelty since they are distinct baseline scenarios
                provenance={"source": "manual", "parent_attack_id": None, "mutation_operators": []}
            )
            
            db_scenario = AttackScenarioDB(**scenario.model_dump())
            db.add(db_scenario)
            
        db.commit()
        print(f"Successfully seeded {len(scenarios)} attack scenarios!")
    finally:
        db.close()

if __name__ == "__main__":
    seed_database()