import asyncio
import logging
import os
import uuid
from typing import Any

from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.sessions import VertexAiSessionService
from tenacity import retry, retry_if_exception, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.exceptions import ConcurrentWriteError, SessionRecoveryError
from src.state_models import NegotiationState

logger = logging.getLogger(__name__)

# Nomes usados na API de sessões do ADK (app_name / user_id)
APP_NAME = "agente-3-the-memory"
USER_ID = "default"

# Política de retry: Exponential Backoff para throttling (429) e falhas transitórias
RETRY_POLICY = dict(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)


def _session_state_only(session: Any) -> dict:
    """Extrai apenas o state de sessão (sem prefixos app:, user:, temp:)."""
    state = getattr(session, "state", None) or {}
    return {
        k: v
        for k, v in state.items()
        if not k.startswith(("app:", "user:", "temp:"))
    }


class NegotiationSessionGateway:
    """
    Abstração Resiliente para Checkpointing e Short-Term Memory.
    Protege a aplicação se a conexão com o Vertex Session Service falhar, e
    garante a validação do estado usando Pydantic.
    Suporta retry com Exponential Backoff e OCC (Optimistic Concurrency Control).
    Compatível com a API do Google ADK (get_session/create_session com app_name, user_id; append_event para persistir state).
    """

    def __init__(self, project_id: str | None = None, location: str | None = None):
        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self.location = location or os.environ.get("GOOGLE_CLOUD_LOCATION") or os.environ.get("GOOGLE_CLOUD_REGION")
        use_vertex_session = os.environ.get("USE_VERTEX_SESSION", "").strip().lower() in ("1", "true")

        self.is_mock = not (bool(self.project_id) and use_vertex_session)
        if self.is_mock:
            from google.adk.sessions import InMemorySessionService

            self.service = InMemorySessionService()
            if self.project_id:
                logger.warning(
                    "Sessao em memoria (USE_VERTEX_SESSION nao ativo). Vertex AI apenas para o modelo LLM."
                )
            else:
                logger.warning("GOOGLE_CLOUD_PROJECT nao definido. Usando Session Gateway mock em memoria.")
        else:
            self.service = VertexAiSessionService(self.project_id, self.location)

    @retry(**RETRY_POLICY)
    def _recover_or_create_sync(self, session_id: str, tier: str) -> tuple[Any, NegotiationState]:
        """Lógica síncrona com retry; chamada via asyncio.to_thread a partir de recover_or_create."""
        try:
            session = self.service.get_session_sync(
                app_name=APP_NAME,
                user_id=USER_ID,
                session_id=session_id,
            )
        except Exception as e:
            raise SessionRecoveryError(f"Falha ao recuperar Checkpoint ADK para {session_id}: {str(e)}") from e

        if session is None:
            state = NegotiationState(funnel_stage="initial_contact", customer_tier=tier)
            try:
                session = self.service.create_session_sync(
                    app_name=APP_NAME,
                    user_id=USER_ID,
                    session_id=session_id,
                    state=state.model_dump(),
                )
            except Exception as e:
                raise SessionRecoveryError(f"Falha ao criar sessão ADK para {session_id}: {str(e)}") from e
            return session, state

        session_state = _session_state_only(session)
        if not session_state:
            state = NegotiationState(funnel_stage="initial_contact", customer_tier=tier)
        else:
            try:
                state = NegotiationState(**session_state)
            except Exception as e:
                logger.error("Checkpoint corrompido! Resetando. Erro: %s", e)
                state = NegotiationState(funnel_stage="initial_contact", customer_tier=tier)

        return session, state

    async def recover_or_create(self, session_id: str, tier: str = "standard") -> tuple[Any, NegotiationState]:
        """Recupera sessão (ou cria) de forma não bloqueante, com retry em caso de falha de rede."""
        if self.is_mock:
            return await asyncio.to_thread(self._recover_or_create_sync, session_id, tier)
        # Vertex: só API async
        try:
            session = await self.service.get_session(
                app_name=APP_NAME, user_id=USER_ID, session_id=session_id
            )
        except Exception as e:
            raise SessionRecoveryError(f"Falha ao recuperar Checkpoint ADK para {session_id}: {str(e)}") from e
        if session is None:
            state = NegotiationState(funnel_stage="initial_contact", customer_tier=tier)
            session = await self.service.create_session(
                app_name=APP_NAME, user_id=USER_ID, state=state.model_dump()
            )
            return session, state
        session_state = _session_state_only(session)
        if not session_state:
            state = NegotiationState(funnel_stage="initial_contact", customer_tier=tier)
        else:
            try:
                state = NegotiationState(**session_state)
            except Exception as e:
                logger.error("Checkpoint corrompido! Resetando. Erro: %s", e)
                state = NegotiationState(funnel_stage="initial_contact", customer_tier=tier)
        return session, state

    def _occ_check_and_bump_sync(self, session: Any, state: NegotiationState) -> None:
        """Verifica OCC e incrementa versão; falha com ConcurrentWriteError se houver conflito."""
        session_id = getattr(session, "id", None) or getattr(session, "session_id", None)
        try:
            current = (
                self.service.get_session_sync(
                    app_name=APP_NAME,
                    user_id=USER_ID,
                    session_id=session_id,
                )
                if session_id
                else session
            )
        except Exception:
            current = session
        if current is None:
            raise SessionRecoveryError(f"Sessão {session_id} não encontrada ao salvar checkpoint.")
        current_state = _session_state_only(current)
        current_version = current_state.get("version", 1)
        if current_version != state.version:
            raise ConcurrentWriteError(
                f"OCC conflict: expected version {state.version}, found {current_version}. "
                "Outro writer persistiu o checkpoint."
            )
        state.bump_version()

    def _save_checkpoint_sync_retry(self, session: Any, state: NegotiationState) -> None:
        """Executa OCC check e bump; o persist real é feito via append_event no save_checkpoint async."""
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception(lambda e: not isinstance(e, ConcurrentWriteError)),
            reraise=True,
        )
        def _do():
            self._occ_check_and_bump_sync(session, state)

        _do()

    async def save_checkpoint(self, session: Any, state: NegotiationState) -> None:
        """Salva a FSM atualizada (OCC) via append_event com state_delta."""
        if self.is_mock:
            try:
                await asyncio.to_thread(self._save_checkpoint_sync_retry, session, state)
            except ConcurrentWriteError:
                raise
            except Exception as e:
                logger.error("CRÍTICO: Falha ao salvar checkpoint ADK após retries. Causa: %s", e)
                raise
        else:
            session_id = getattr(session, "id", None) or getattr(session, "session_id", None)
            current = await self.service.get_session(
                app_name=APP_NAME, user_id=USER_ID, session_id=session_id
            )
            if current is None:
                raise SessionRecoveryError(f"Sessão {session_id} não encontrada ao salvar checkpoint.")
            current_state = _session_state_only(current)
            current_version = current_state.get("version", 1)
            if current_version != state.version:
                raise ConcurrentWriteError(
                    f"OCC conflict: expected version {state.version}, found {current_version}."
                )
            state.bump_version()

        event = Event(
            author="StatefulFinanceAgent",
            invocation_id=str(uuid.uuid4()),
            actions=EventActions(state_delta=state.model_dump()),
        )
        await self.service.append_event(session=session, event=event)
