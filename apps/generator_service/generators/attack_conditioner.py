import pandas as pd
import numpy as np
import httpx
from uuid import uuid4
from datetime import timedelta
import os

# ... inside fetch_attack_scenario or at the top of the file:
THREAT_SERVICE_URL = os.getenv("THREAT_SERVICE_URL", "http://localhost:8001")

def fetch_attack_scenario(attack_id: str) -> dict:
    """Calls Member 1 API to get attack parameters."""
    try:
        # Using context manager to ensure the socket is closed after the request
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{THREAT_SERVICE_URL}/api/v1/attacks/{attack_id}")
            response.raise_for_status()
            envelope = response.json()
            return envelope.get("data", {})
    except Exception as e:
        print(f"Warning: Could not fetch attack {attack_id} from threat_service. Error: {e}")
        return None

def apply_attack_distortions(df: pd.DataFrame, scenario: dict, existing_customers: list) -> pd.DataFrame:
    if not scenario or "parameters" not in scenario:
        return df
        
    params = scenario["parameters"]
    attack_id = scenario["attack_id"]
    
    df = df.copy()
    df["is_fraud"] = True
    df["attack_id"] = attack_id
    
    # CRITICAL FIX: Use existing customers so fraud rows have prior history!
    df["customer_id"] = np.random.choice(existing_customers, size=len(df))
    
    # 1. MULE/Shared Device Logic
    if params.get("shared_device", False):
        # Assign ONE device to all rows in this batch
        shared_dev = f"DEV_SHARED_{uuid4().hex[:8]}"
        df["device_id"] = shared_dev
        
    # 2. ATO/New Device Logic
    elif params.get("new_device", False):
        # Assign a new device to every row
        df["device_id"] = [f"DEV_NEW_{uuid4().hex[:8]}" for _ in range(len(df))]
        
    # 3. Velocity Multiplier
    vel_mult = params.get("velocity_multiplier", 1)
    if vel_mult > 1:
        time_delta = timedelta(minutes=30 / vel_mult)
        base_time = df["timestamp"].min()
        df["timestamp"] = [base_time + (time_delta * i) for i in range(len(df))]
        
    # 4. Amount Anomaly
    amount_z = params.get("amount_anomaly_z", 0.0)
    if amount_z != 0.0:
        mean_amt = df["amount"].mean()
        std_amt = df["amount"].std()
        if std_amt > 0:
            df["amount"] = mean_amt + (amount_z * std_amt) + np.random.normal(0, std_amt * 0.15, size=len(df))
            
    # 5. Merchant Novelty
    if params.get("merchant_novelty", False):
        df["merchant_id"] = [f"M_UNKNOWN_{np.random.randint(1000, 1999)}" for _ in range(len(df))]
        
    return df

def generate_attack_transactions(attack_id: str, rows: int, seed: int, existing_customers: list) -> pd.DataFrame:
    from apps.generator_service.generators.baseline import generate_baseline_transactions
    df_baseline = generate_baseline_transactions(rows=rows, seed=seed)
    
    scenario = fetch_attack_scenario(attack_id)
    if not scenario:
        raise ValueError(f"Attack scenario {attack_id} not found.")
        
    # Pass existing_customers to the distorter!
    df_fraud = apply_attack_distortions(df_baseline, scenario, existing_customers)
    return df_fraud