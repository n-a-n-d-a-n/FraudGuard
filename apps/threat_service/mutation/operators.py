from shared.schemas.attack import AttackParameters

def apply_mutations(original_params: AttackParameters, operators: list[dict]) -> tuple[AttackParameters, list[str]]:
    """
    Applies mutation operators to attack parameters.
    Returns the new parameters and a list of applied operator names.
    """
    # Create a mutable copy
    new_params = original_params.model_copy()
    applied_ops = []
    
    for op in operators:
        name = op.get("name")
        params = op.get("params", {})
        
        if name == "bump_velocity":
            # Increase velocity multiplier
            new_mult = new_params.velocity_multiplier + params.get("add_multiplier", 2)
            new_params.velocity_multiplier = max(1, new_mult)
            applied_ops.append(f"bump_velocity(+{params.get('add_multiplier', 2)})")
            
        elif name == "shift_geography":
            # Force a location shift
            new_params.location_shift = True
            applied_ops.append("shift_geography(forced)")
            
        elif name == "swap_device":
            # Force a new device
            new_params.new_device = True
            applied_ops.append("swap_device(forced)")
            
        elif name == "increase_amount_z":
            # Make the amount anomaly larger
            new_z = new_params.amount_anomaly_z + params.get("add_z", 1.0)
            new_params.amount_anomaly_z = new_z
            applied_ops.append(f"increase_amount_z(+{params.get('add_z', 1.0)})")
            
    return new_params, applied_ops