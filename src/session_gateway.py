import asyncio
import logging
import os
from typing import Any

from google_adk.sessions import VertexAiSessionService
from tenacity import retry, retry_if_exception, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.exceptions import ConcurrentWriteError, SessionRecoveryError
from src.state_models import NegotiationState

logger = logging.getLogger(__name__)

# Política de retry: Exponential Backoff para throttling (429) e falhas transitórias
RETRY_POLICY = dict(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)


class NegotiationSessionGateway:
    """
    Abstração Resiliente para Checkpointing e Short-Term Memory.
    Protege a aplicação se a conexão com o Vertex Session Service falhar, e
    garante a validação do estado usando Pydantic.
    Suporta retry com Exponential Backoff e OCC (Optimistic Concurrency Control).
    """

    def __init__(self, project_id: str | None = None, location: str | None = None):
        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self.location = location or os.environ.get("GOOGLE_CLOUD_REGION")

        self.is_mock = not bool(self.project_id)
        if not self.is_mock:
            self.service = VertexAiSessionService(self.project_id, self.location)
        else:
            logger.warning("GOOGLE_CLOUD_PROJECT não definido. Usando Session Gateway mock em memória.")
            from google_adk.sessions import InMemorySessionService

            self.service = InMemorySessionService()

    @retry(**RETRY_POLICY)
    def _recover_or_create_sync(self, session_id: str, tier: str) -> tuple[Any, NegotiationState]:
        """Lógica síncrona com retry; chamada via asyncio.to_thread a partir de recover_or_create."""
        try:
            session = self.service.get_session(session_id)
        except Exception as e:
            raise SessionRecoveryError(f"Falha ao recuperar Checkpoint ADK para {session_id}: {str(e)}") from e

        if not session.state:
            state = NegotiationState(funnel_stage="initial_contact", customer_tier=tier)
            session.state = state.model_dump()
        else:
            try:
                state = NegotiationState(**session.state)
            except Exception as e:
                logger.error(f"Checkpoint corrompido! Resetando. Erro: {e}")
                state = NegotiationState(funnel_stage="initial_contact", customer_tier=tier)
                session.state = state.model_dump()

        return session, state

    async def recover_or_create(self, session_id: str, tier: str = "standard") -> tuple[Any, NegotiationState]:
        """Recupera sessão (ou cria) de forma não bloqueante, com retry em caso de falha de rede."""
        return await asyncio.to_thread(self._recover_or_create_sync, session_id, tier)

    def _save_checkpoint_sync(self, session: Any, state: NegotiationState) -> None:
        """Salva checkpoint com OCC: verifica versão antes de escrever."""
        session_id = getattr(session, "id", None) or getattr(session, "session_id", None)
        try:
            current = self.service.get_session(session_id) if session_id else session
        except Exception:
            current = session
        current_state = current.state or {}
        current_version = current_state.get("version", 1)
        if current_version != state.version:
            raise ConcurrentWriteError(
                f"OCC conflict: expected version {state.version}, found {current_version}. "
                "Outro writer persistiu o checkpoint."
            )
        state.bump_version()
        session.state = state.model_dump()
        self.service.save_session(session)

    def _save_checkpoint_sync_retry(self, session: Any, state: NegotiationState) -> None:
        """Wrapper que aplica retry em falhas de rede; não retenta em ConcurrentWriteError (OCC)."""

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception(lambda e: not isinstance(e, ConcurrentWriteError)),
            reraise=True,
        )
        def _do_save():
            self._save_checkpoint_sync(session, state)

        _do_save()

    async def save_checkpoint(self, session: Any, state: NegotiationState) -> None:
        """Salva a FSM atualizada (OCC). Não bloqueia o event loop."""
        try:
            await asyncio.to_thread(self._save_checkpoint_sync_retry, session, state)
        except ConcurrentWriteError:
            raise
        except Exception as e:
            logger.error("CRÍTICO: Falha ao salvar checkpoint ADK após retries. Causa: %s", e)
            raise
