import asyncio

import pytest


def _session_gateway():
    """Import opcional: evita Skip no nível do módulo (INTERNALERROR com pytest-asyncio)."""
    pytest.importorskip("google_adk", reason="google-adk não instalado (opcional para testes de gateway)")
    from src.session_gateway import NegotiationSessionGateway
    return NegotiationSessionGateway


def test_session_gateway_recover_or_create_async():
    """Recover_or_create é async e retorna estado com version (OCC)."""
    NegotiationSessionGateway = _session_gateway()
    gw = NegotiationSessionGateway(project_id="", location="")
    session, state = asyncio.run(gw.recover_or_create("test-session-1", "premium"))

    assert state.customer_tier == "premium"
    assert state.funnel_stage == "initial_contact"
    assert state.version == 1


def test_session_gateway_save_checkpoint_async():
    """Save_checkpoint persiste estado e não bloqueia (async)."""
    NegotiationSessionGateway = _session_gateway()
    gw = NegotiationSessionGateway(project_id="", location="")
    session, state = asyncio.run(gw.recover_or_create("test-session-2", "standard"))
    state.increment_rejection(max_rejections=3)

    asyncio.run(gw.save_checkpoint(session, state))

    session2, state2 = asyncio.run(gw.recover_or_create("test-session-2", "standard"))
    assert state2.rejection_count == 1
    assert state2.version >= 1
