import asyncio
import logging
import os

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.exceptions import VectorSearchError

logger = logging.getLogger(__name__)

RETRY_POLICY = dict(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)


class LongTermMemoryGateway:
    """
    Abstração da Memória de Longo Prazo.
    Oculta a complexidade de conexão com o Vertex AI Vector Search.
    Implementa "Graceful Degradation" e retry com Exponential Backoff.
    """

    def __init__(
        self,
        project_id: str | None = None,
        location: str | None = None,
        index_endpoint: str | None = None,
    ):
        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self.location = location or os.environ.get("GOOGLE_CLOUD_REGION")
        self.index_endpoint = index_endpoint or os.environ.get("VECTOR_SEARCH_ENDPOINT_ID")

        self.is_mock = not bool(self.index_endpoint)
        if not self.is_mock:
            from google.adk.memory import VertexAiMemoryBankService

            self.service = VertexAiMemoryBankService(
                project_id=self.project_id,
                location=self.location,
                index_endpoint_id=self.index_endpoint,
            )
        else:
            logger.warning("VECTOR_SEARCH_ENDPOINT_ID não configurado. Usando Mock de Banco Vetorial.")

    def _search_customer_insights_sync(self, query: str) -> str:
        """Lógica síncrona com retry; chamada via asyncio.to_thread."""
        if self.is_mock:
            if "sessao_premium" in query:
                return "O cliente é conservador, negocia as taxas com agressividade e só fecha com taxas < 1.0%."
            return "Nenhum histórico prévio encontrado para este CPF."

        try:
            results = self.service.search_memory(query=query)
            return " ".join([doc.content for doc in results]) if results else ""
        except Exception as e:
            logger.error("Falha na consulta Vetorial. Degrading gracefully... Erro: %s", e)
            raise VectorSearchError(str(e)) from e

    @retry(**RETRY_POLICY)
    def _search_customer_insights_sync_retry(self, query: str) -> str:
        """Busca com retry; após esgotar retries, o caller pode degradar (retornar '')."""
        return self._search_customer_insights_sync(query)

    async def search_customer_insights(self, query: str) -> str:
        """
        Busca conhecimento do cliente (RAG context). Não bloqueia o event loop.
        Em falha persistente após retries, retorna string vazia (graceful degradation).
        """
        try:
            return await asyncio.to_thread(self._search_customer_insights_sync_retry, query)
        except Exception:
            logger.warning("Long-Term Memory indisponível após retries. Degradando para contexto vazio.")
            return ""
