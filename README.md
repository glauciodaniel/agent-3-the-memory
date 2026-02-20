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

## 🛠️ Configuração do Google Cloud e do modelo LLM

O agente usa o **Google ADK** com Gemini. É obrigatório configurar uma das opções abaixo.

### Opção A – Google AI (API Key, ideal para desenvolvimento local)

Crie uma API key em [Google AI Studio](https://aistudio.google.com/apikey) e defina:

```bash
export GOOGLE_API_KEY="sua-api-key"
```

Ou no `.env`:
```
GOOGLE_API_KEY=sua-api-key
```

### Opção B – Vertex AI (projeto GCP, usado na Masterclass)

Para usar Vertex AI com Application Default Credentials:

```bash
gcloud auth application-default login
export GOOGLE_GENAI_USE_VERTEXAI=1
export GOOGLE_CLOUD_PROJECT="banco-auto-finance-lab-01"
export GOOGLE_CLOUD_LOCATION="us-central1"
export VECTOR_SEARCH_ENDPOINT_ID="<ID_FORNECIDO_NA_AULA>"
```

*(Sem variáveis de sessão/Vector Search, o código usa mocks em memória e exibe avisos. Sem API key nem Vertex configurado, o modelo Gemini retorna erro de autenticação.)*

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
