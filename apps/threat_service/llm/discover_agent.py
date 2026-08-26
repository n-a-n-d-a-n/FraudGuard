import os
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, ValidationError
from langchain_groq import ChatGroq
from shared.schemas.attack import (
    AttackScenario, AttackCategory, RiskLevel, AttackParameters, Provenance,
)



class LLMDraft(BaseModel):
    """Everything we ask Llama for. IDs, risk_level and channel stay
    backend-controlled so the LLM can't invent an invalid enum value."""
    description: str
    parameters: AttackParameters
    features: list[str]

class AgentState(TypedDict):
    category: str
    draft: Optional[dict]
    validated_scenario: Optional[dict]
    error: Optional[str]

MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# Constructing this is lazy/cheap — it does NOT hit the network or require
# a key until .invoke() is actually called, so importing this module is safe.
_llm = ChatGroq(model=MODEL_NAME, temperature=0.7)
_structured_llm = _llm.with_structured_output(LLMDraft)

PROMPT_TEMPLATE = """You are a payment-fraud threat-intelligence analyst supporting a \
hackathon red-team simulation exercise (Mastercard Innovation Challenge).

Propose ONE plausible, controlled, synthetic payment-fraud scenario in the \
"{category}" family.

Rules:
- This is for generating SYNTHETIC training/test data only. Never reference real \
people, real account numbers, real credentials, or step-by-step fraud instructions.
- Ground it in observable transaction/behavioral signals (velocity, device change, \
geography shift, amount anomaly, merchant novelty) rather than operational detail.
- description: 1-2 sentences, written like a threat-intel note.
- features: short signal names such as "velocity_spike", "new_device", \
"location_shift", "merchant_novelty" — NOT raw parameter names.
- parameters: realistic values for new_device, velocity_multiplier, location_shift, \
amount_anomaly_z, time_window_minutes, merchant_novelty given this attack family.
"""

def generate_idea(state: AgentState):
    """Node 1: call Llama 3.3 for a structured scenario draft."""
    category = state["category"]
    try:
        draft: LLMDraft = _structured_llm.invoke(
            PROMPT_TEMPLATE.format(category=category)
        )
        return {"draft": draft.model_dump(), "error": None}
    except Exception as e:
        # Covers auth errors, rate limits, and any structured-output parse failure
        return {"draft": None, "error": f"LLM generation failed: {e}"}

def validate_json(state: AgentState):
    """Node 2: assemble + validate the full AttackScenario with Pydantic."""
    draft = state.get("draft")
    if draft is None:
        return {"error": state.get("error") or "No draft produced"}

    try:
        category_enum = AttackCategory(state["category"])
        scenario = AttackScenario(
            attack_id=f"{category_enum.value[:4]}_LLM_{os.urandom(3).hex()}",
            category=category_enum,
            channel="CARD",
            risk_level=RiskLevel.HIGH,
            description=draft["description"],
            parameters=AttackParameters(**draft["parameters"]),
            features=draft["features"],
            novelty_score=0.85,
            provenance=Provenance(source="llm_discovery"),
        )
        return {"validated_scenario": scenario.model_dump(), "error": None}
    except (ValidationError, KeyError, ValueError) as e:
        return {"error": f"Validation failed: {e}"}

# Build the Graph
workflow = StateGraph(AgentState)
workflow.add_node("generate_idea", generate_idea)
workflow.add_node("validate_json", validate_json)

workflow.set_entry_point("generate_idea")
workflow.add_edge("generate_idea", "validate_json")
workflow.add_edge("validate_json", END)

discover_chain = workflow.compile()