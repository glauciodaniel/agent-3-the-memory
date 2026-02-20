import pytest
from pydantic import ValidationError
from src.state_models import NegotiationState


def test_negotiation_state_initialization():
    """Valida se o estado inicializa corretamente com valores padrão seguros."""
    state = NegotiationState(funnel_stage="initial_contact", customer_tier="standard")

    assert state.funnel_stage == "initial_contact"
    assert state.rejection_count == 0
    assert state.proposed_rate is None
    assert state.customer_tier == "standard"
    assert state.version == 1


def test_fsm_circuit_breaker_default_max_rejections():
    """Valida a transição da Máquina de Estados no Circuit Breaker (max_rejections=3)."""
    state = NegotiationState(funnel_stage="rate_proposed", customer_tier="premium")

    state.increment_rejection(max_rejections=3)
    state.increment_rejection(max_rejections=3)
    assert state.rejection_count == 2
    assert state.funnel_stage == "rate_proposed"

    state.increment_rejection(max_rejections=3)
    assert state.rejection_count == 3
    assert state.funnel_stage == "human_handoff"


def test_fsm_circuit_breaker_configurable_max_rejections():
    """max_rejections vindo de config (ex: memory_policy.yaml) ativa handoff no limite."""
    state = NegotiationState(funnel_stage="rate_proposed", customer_tier="premium")

    state.increment_rejection(max_rejections=2)
    assert state.rejection_count == 1
    assert state.funnel_stage == "rate_proposed"

    state.increment_rejection(max_rejections=2)
    assert state.rejection_count == 2
    assert state.funnel_stage == "human_handoff"


def test_state_version_bump():
    """OCC: bump_version incrementa versão para detectar race conditions."""
    state = NegotiationState(funnel_stage="initial_contact", customer_tier="standard")
    assert state.version == 1
    state.bump_version()
    assert state.version == 2
    state.bump_version()
    assert state.version == 3


def test_invalid_funnel_stage_rejected():
    """Pydantic deve bloquear um estágio de funil inventado pelo LLM."""
    with pytest.raises(ValidationError):
        NegotiationState(funnel_stage="fase_inventada_pelo_llm", customer_tier="premium")
