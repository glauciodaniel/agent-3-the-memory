"""
Gera o PDF do lab RAG (Conta Premium) em data/lab/lab_conta_premium.pdf.
Uso: python scripts/generate_lab_pdf.py
Requer: pip install fpdf2  ou  pip install -e ".[lab]"
"""

from pathlib import Path


def main() -> None:
    try:
        from fpdf import FPDF
    except ImportError as e:
        raise SystemExit(
            "fpdf2 não instalado. Rode: pip install fpdf2  ou  pip install -e \".[lab]\""
        ) from e

    out_dir = Path(__file__).resolve().parent.parent / "data" / "lab"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "lab_conta_premium.pdf"

    pdf = FPDF()
    pdf.set_margins(20, 20, 20)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=14)
    w = pdf.epw  # effective page width
    pdf.cell(w, 10, "Conta Premium - Produto Banco (Lab RAG)", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", size=11)

    # Texto em ASCII para evitar problemas de encoding com fpdf2/Helvetica
    body = """
A Conta Premium e um produto para clientes com maior relacionamento e renda. Beneficios:

- Isencao de taxa de manutencao mensal.
- TEDs: ate 5 gratuitos por mes; apos isso, R$ 15,00 por TED.
- Saques em rede propria: ilimitados. Saques em rede 24h: R$ 10,00 cada.
- Cartao de debito e credito sem anuidade no primeiro ano; segunda via a R$ 35,00.
- Linha de credito pre-aprovada e taxas diferenciadas em emprestimo pessoal (a partir de 0,85% a.m.).

Requisitos para adesao:
- Renda mensal comprovada minima: R$ 5.000.
- Manter saldo medio de R$ 10.000 ou aplicacoes equivalentes no trimestre.

Canais de atendimento:
- App, internet banking, agencia e central de relacionamento (telefone). Proposta de emprestimo pode ser feita pelo agente virtual; condicoes sujeitas a analise de credito.
"""
    for line in body.strip().split("\n"):
        line = line.strip()
        if not line:
            pdf.ln(5)
            continue
        pdf.multi_cell(w, 6, line)
    pdf.ln(5)
    pdf.set_font("Helvetica", size=10)
    pdf.cell(w, 6, "Documento gerado por scripts/generate_lab_pdf.py para o lab RAG ChromaDB.", new_x="LMARGIN", new_y="NEXT")

    pdf.output(str(out_path))
    print(f"PDF gerado: {out_path}")


if __name__ == "__main__":
    main()
