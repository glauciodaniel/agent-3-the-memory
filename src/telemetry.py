import tiktoken
import yaml


class FinOpsTelemetry:
    """
    Calcula e compara o custo do modelo Arquitetural Stateful (ADK)
    versus a abordagem Stateless/Amnésica (onde se envia o chat history inteiro).
    """

    def __init__(self, config_path: str = "config/memory_policy.yaml"):
        with open(config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)["finops"]

        # Encodings do tiktoken como proxy de tokens Gemini
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.encoding = None

    def calculate_stateful_cost(self, prompt_text: str, response_text: str) -> float:
        """Calcula custo do modelo arquiteturado com Checkpointing + RAG vetorial."""
        input_tokens = len(self.encoding.encode(prompt_text)) if self.encoding else len(prompt_text) // 4
        output_tokens = len(self.encoding.encode(response_text)) if self.encoding else len(response_text) // 4

        cost_in = (input_tokens / 1000) * self.config["cost_per_1k_input"]
        cost_out = (output_tokens / 1000) * self.config["cost_per_1k_output"]
        return cost_in + cost_out

    def get_amnesic_baseline_cost(self) -> float:
        """Custo hipotético de enviar o histórico completo (10k tokens)."""
        tokens = self.config["amnesic_payload_tokens"]
        return (tokens / 1000) * self.config["cost_per_1k_input"]

    def print_savings_report(self, stateful_cost: float):
        """Imprime relatório FinOps comparativo no console."""
        amnesic = self.get_amnesic_baseline_cost()
        savings = amnesic - stateful_cost
        pct = (savings / amnesic) * 100 if amnesic > 0 else 0

        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="💰 FinOps: Custo de Memória (Por Request)")
        table.add_column("Abordagem", justify="left")
        table.add_column("Custo USD", justify="right")
        table.add_column("Tokens Estimados", justify="right")

        table.add_row("❌ Amnésico (Histórico Completo)", f"${amnesic:.6f}", str(self.config["amnesic_payload_tokens"]))
        table.add_row("✅ Stateful (ADK Checkpoint + Vetorial)", f"${stateful_cost:.6f}", "~500")
        table.add_row(
            "[bold green]Economia (FinOps)[/bold green]", f"[bold green]${savings:.6f} (-{pct:.1f}%)[/bold green]", ""
        )

        console.print(table)
