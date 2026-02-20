import asyncio
import logging
from pathlib import Path

import jinja2
import yaml
from google_adk import LlmAgent

from src.memory_gateway import LongTermMemoryGateway
from src.session_gateway import NegotiationSessionGateway

logger = logging.getLogger(__name__)

# Caminho padrão da política (relativo ao CWD do processo)
DEFAULT_POLICY_PATH = Path("config/memory_policy.yaml")


def _load_max_rejections(config_path: Path | None = None) -> int:
    path = config_path or DEFAULT_POLICY_PATH
    if not path.exists():
        return 3
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return (data.get("session") or {}).get("max_rejections", 3)


class StatefulFinanceAgent:
    """
    Agente Mestre orquestrador de memória.
    Integra LlmAgent do Google ADK com os Gateways de curto e longo prazo.
    Dependências injetadas (IoC) para testes e substituição de infraestrutura.
    """

    def __init__(
        self,
        session_gw: NegotiationSessionGateway,
        memory_gw: LongTermMemoryGateway,
        *,
        policy_path: Path | None = None,
    ):
        self.session_gw = session_gw
        self.memory_gw = memory_gw
        self._max_rejections = _load_max_rejections(policy_path)

        self.jinja_env = jinja2.Environment(loader=jinja2.FileSystemLoader("prompts"))
        self.negotiator_template = self.jinja_env.get_template("negotiator.jinja2")
        self.injector_template = self.jinja_env.get_template("context_injector.jinja2")

        self.llm_agent = LlmAgent(
            name="StatefulAutoFinanceNegotiator",
            model="gemini-1.5-pro-001",
            system_instruction="Você é um negociador de financiamentos. O contexto será dinamicamente injetado.",
        )

    async def process_message(
        self,
        session_id: str,
        customer_message: str,
        customer_tier: str = "standard",
    ) -> str:
        """Fluxo orquestrado (async): injeta estado e memória vetorial no prompt."""
        adk_session, state = await self.session_gw.recover_or_create(session_id, customer_tier)

        if state.funnel_stage == "human_handoff":
            return "[SYSTEM] Negociação encerrada pelo agente. Aguarde a transferência para um especialista."

        system_prompt = self.negotiator_template.render(
            funnel_stage=state.funnel_stage,
            proposed_rate=state.proposed_rate,
            rejection_count=state.rejection_count,
            customer_tier=state.customer_tier,
        )
        self.llm_agent.system_instruction = system_prompt

        contextual_prompt = customer_message
        if state.funnel_stage in ("rate_proposed", "analyzing_credit"):
            insights = await self.memory_gw.search_customer_insights(query=f"Sessão: {session_id}")
            if insights:
                contextual_prompt = self.injector_template.render(
                    base_prompt=customer_message,
                    long_term_insights=insights,
                )

        response = await asyncio.to_thread(
            self.llm_agent.invoke,
            contextual_prompt,
            session=adk_session,
        )

        state.increment_rejection(max_rejections=self._max_rejections)
        if state.funnel_stage == "human_handoff":
            pass  # Handoff já refletido no state
        await self.session_gw.save_checkpoint(adk_session, state)

        return response.text
