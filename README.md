# Agente 3: The Memory

Projeto didático e arquitetural para ensino de construção de Agentes de IA Stateful utilizando "Defense in Depth" com Máquinas de Estados (FSM) tipadas (Pydantic), FinOps em tokens e Checkpointing utilizando **Google ADK**. 

## 📋 Arquitetura de Estado (Stateful Agent Pattern)

Este laboratório rompe com o anti-pattern do "Agente Amnésico" que sempre recarrega todo o histórico a cada chamada (onerando tokens). Aqui nós dividimos a persistência em duas camadas orquestradas por **Gateways**:

1. **Short-Term Memory (Session Gateway)**: 
   Salva o estado FSM *atual* da negociação no Vertex AI Session Service. Retém contadores (rejection_count), estágio do funil e política do cliente.
2. **Long-Term Memory (Memory Gateway)**:
   Acesso pontual (apenas quando justificado pelo funil) a um Vector Search do GCP para injetar no System Prompt insights prévios do relacionamento com o Banco.

## 🚀 Instalação (Standalone)

```bash
# Clone ou acesse este diretório de Lab
cd agente-3-the-memory

# Ambiente virtual isolado
python -m venv venv
# Linux/Mac
source venv/bin/activate
# Windows
venv\Scripts\activate

# Instalação das bibliotecas e ADK
pip install -r requirements.txt
```

## 🛠️ Configuração do Google Cloud (Necessário na Masterclass)
Para operar contra a infraestrutura do Google Vertex AI provida para o Lab:

```bash
gcloud auth login --update-adc
export GOOGLE_CLOUD_PROJECT="banco-auto-finance-lab-01"
export GOOGLE_CLOUD_REGION="us-central1"
export VECTOR_SEARCH_ENDPOINT_ID="<ID_FORNECIDO_NA_AULA>"
```

*(Nota: Se rodado sem as variáveis de ambiente, o código usará Mocks em memória e exibirá alertas.)*

## 🧑‍💻 Execução Local e Testes

Para rodar a demonstração arquitetural orquestrada no `main.py`:
```bash
python -m src.main
```

Para validar as políticas de Estado com `pytest`:
```bash
pytest tests/ -v
```

## 📚 Documentação Adicional
Consulte o arquivo [LAB-DESAFIO.md](LAB-DESAFIO.md) para as instruções do desafio hand-on e detalhes de FinOps.
