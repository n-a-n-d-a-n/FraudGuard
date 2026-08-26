import pytest
from pydantic import ValidationError
from shared.schemas.attack import AttackScenario, AttackParameters, AttackCategory, RiskLevel
from apps.threat_service.mutation.operators import apply_mutations

def test_valid_attack_scenario():
    """Test that a perfectly structured attack scenario passes validation."""
    scenario = AttackScenario(
        attack_id="TEST_001",
        category="ACCOUNT_TAKEOVER",
        channel="CARD",
        risk_level="HIGH",
        description="A test scenario",
        parameters={"new_device": True, "velocity_multiplier": 3},
        features=["new_device"]
    )
    assert scenario.attack_id == "TEST_001"
    assert scenario.parameters.new_device is True

def test_invalid_attack_category():
    """Test that Pydantic rejects an invalid attack category."""
    with pytest.raises(ValidationError):
        AttackScenario(
            attack_id="TEST_002",
            category="NOT_A_REAL_CATEGORY", # Should fail validation
            channel="CARD",
            risk_level="HIGH",
            description="Bad scenario",
            parameters={},
            features=[]
        )

def test_mutation_bump_velocity():
    """Test that the bump_velocity mutation operator correctly increases velocity."""
    original_params = AttackParameters(velocity_multiplier=2)
    
    # Apply the mutation: add 3 to the velocity
    new_params, applied_ops = apply_mutations(original_params, [{"name": "bump_velocity", "params": {"add_multiplier": 3}}])
    
    assert new_params.velocity_multiplier == 5  # 2 + 3 = 5
    assert len(applied_ops) == 1
    assert "bump_velocity" in applied_ops[0]

def test_mutation_swap_device():
    """Test that the swap_device mutation operator forces new_device to True."""
    original_params = AttackParameters(new_device=False)
    
    new_params, applied_ops = apply_mutations(original_params, [{"name": "swap_device"}])
    
    assert new_params.new_device is True