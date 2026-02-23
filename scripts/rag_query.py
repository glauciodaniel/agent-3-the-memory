"""
Consulta RAG (memória de longo prazo): envia uma pergunta e imprime o contexto recuperado.
Uso: python scripts/rag_query.py "Qual a tarifa da conta premium e condições de empréstimo?"
Requer que o projeto esteja instalado (pip install -e .) e config/memory_policy.yaml com backend (chroma | vertex | mock).
Carrega variáveis do .env da raiz do projeto (GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION) se existir.
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Raiz do repositório
repo_root = Path(__file__).resolve().parent.parent

# Carregar .env da raiz (GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, etc.) antes de importar src
try:
    from dotenv import load_dotenv
    load_dotenv(repo_root / ".env", override=True)
except ImportError:
    pass

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.memory_gateway import LongTermMemoryGateway


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Consulta o vector store (RAG) e imprime o contexto recuperado para uma pergunta."
    )
    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        default="Quais são as tarifas da conta premium e as condições para empréstimo pessoal?",
        help="Pergunta para buscar no RAG (default: pergunta padrão do lab)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=repo_root / "config" / "memory_policy.yaml",
        help="Caminho do memory_policy.yaml",
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"Erro: arquivo de config não encontrado: {args.config}", file=sys.stderr)
        return 1

    gateway = LongTermMemoryGateway(config_path=args.config)
    result = asyncio.run(gateway.search_customer_insights(args.query))

    print(result if result else "(nenhum trecho recuperado)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
