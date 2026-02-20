import asyncio

from src.memory_gateway import LongTermMemoryGateway


def test_ltm_gateway_graceful_degradation():
    """Valida se o gateway Vetorial retorna mock quando em modo fallback (async)."""
    gw = LongTermMemoryGateway(project_id="", location="", index_endpoint="")

    res = asyncio.run(gw.search_customer_insights("query: teste para sessao_premium"))
    assert "cliente é conservador" in res

    res = asyncio.run(gw.search_customer_insights("query_que_nao_da_match"))
    assert res == "Nenhum histórico prévio encontrado para este CPF."
