import asyncio
import logging

from src.agent_router import StatefulFinanceAgent
from src.memory_gateway import LongTermMemoryGateway
from src.session_gateway import NegotiationSessionGateway
from src.telemetry import FinOpsTelemetry

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


async def run_lab_async() -> None:
    print("=" * 60)
    print("🚗 Agente 3: The Memory - Demonstração FSM e FinOps")
    print("=" * 60)

    session_id = "sessao_premium_998877"

    session_gw = NegotiationSessionGateway()
    memory_gw = LongTermMemoryGateway()
    agent = StatefulFinanceAgent(session_gw=session_gw, memory_gw=memory_gw)
    telemetry = FinOpsTelemetry()

    messages = [
        "Olá, gostaria de financiar um SUV elétrico de R$ 350.000.",
        "A taxa que vocês ofereceram ontem está muito alta. Não aceito 1.49%.",
        "Ainda acho alto. Não vou fechar o financiamento assim.",
    ]

    print(f"\n[SISTEMA] Iniciando recuperação de Checkpoint (Sessão: {session_id})...")

    total_cost = 0.0
    for i, msg in enumerate(messages, 1):
        print(f"\n[Turno {i}]")
        print(f"👤 Cliente: {msg}")

        response_text = await agent.process_message(
            session_id=session_id,
            customer_message=msg,
            customer_tier="premium",
        )

        print(f"🤖 Agente: {response_text}")
        cost = telemetry.calculate_stateful_cost(msg, response_text)
        total_cost += cost

    print("\n")
    telemetry.print_savings_report(total_cost)


def run_lab() -> None:
    asyncio.run(run_lab_async())


if __name__ == "__main__":
    run_lab()
