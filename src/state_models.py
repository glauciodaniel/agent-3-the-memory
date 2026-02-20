from typing import Literal

from pydantic import BaseModel, Field


class NegotiationState(BaseModel):
    """
    Modelo estrito para a Máquina de Estados (FSM) da Negociação.
    Evita corrupção do checkpoint (Short-Term Memory) pelo LLM.
    Campo `version` permite Optimistic Concurrency Control (OCC) no save.
    """

    funnel_stage: Literal["initial_contact", "analyzing_credit", "rate_proposed", "contract_signed", "human_handoff"]
    rejection_count: int = Field(default=0, ge=0)
    proposed_rate: float | None = None
    customer_tier: Literal["standard", "premium"]
    version: int = Field(default=1, ge=1, description="OCC: incrementado a cada save para detectar race conditions")

    def increment_rejection(self, max_rejections: int = 3) -> None:
        """Incrementa contador de recusa. Handoff quando atingir max_rejections (configurável via YAML)."""
        self.rejection_count += 1
        if self.rejection_count >= max_rejections:
            self.funnel_stage = "human_handoff"

    def bump_version(self) -> None:
        """Incrementa versão para OCC; chamado antes de persistir."""
        self.version += 1
