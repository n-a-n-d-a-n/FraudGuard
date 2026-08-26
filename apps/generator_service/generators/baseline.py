import pandas as pd
import numpy as np
from uuid import uuid4
from datetime import datetime, timezone, timedelta

def generate_baseline_transactions(rows: int, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    
    # Generate synthetic customers and merchants
    customer_ids = [f"C{str(i).zfill(5)}" for i in np.random.randint(1, 101, rows)]
    merchant_ids = [f"M{str(i).zfill(4)}" for i in np.random.randint(1, 501, rows)]
    
    # Generate realistic amounts (log-normal distribution)
    amounts = np.random.lognormal(mean=3.5, sigma=0.8, size=rows).round(2)
    
    # Generate timestamps within the last 30 days
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=30)
    timestamps = pd.date_range(start=start_time, end=end_time, periods=rows).to_pydatetime()
    
    # Generate channels (weighted distribution)
    channels = np.random.choice(["CARD", "UPI", "WALLET", "P2P"], size=rows, p=[0.5, 0.3, 0.15, 0.05])
    
    # Generate device IDs (90% sticky to customer, 10% new device)
    device_ids = []
    for cust_id in customer_ids:
        if np.random.rand() < 0.9:
            # Sticky device (derive from customer ID for consistency)
            device_ids.append(f"DEV_{cust_id[1:]}")
        else:
            # New device
            device_ids.append(f"DEV_{uuid4().hex[:8]}")
    
    # Assemble DataFrame
    df = pd.DataFrame({
        "transaction_id": [f"TX_{uuid4().hex[:12]}" for _ in range(rows)],
        "customer_id": customer_ids,
        "merchant_id": merchant_ids,
        "amount": amounts,
        "currency": "USD",
        "timestamp": timestamps,
        "channel": channels,
        "device_id": device_ids,
        "merchant_category": np.random.choice(["RETAIL", "FOOD", "TRAVEL", "DIGITAL", "GROCERY"], size=rows),
        "attack_id": None,
        "is_fraud": False
    })
    
    return df