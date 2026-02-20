# 🧑‍💻 Hands-On: Desafio "The Memory"

## O Problema (Amnésia e FinOps)
Ao analisar logs da operação, notamos que o banco tem gasto milhares de dólares em tokens Gemini simplesmente re-enviando TODO o histórico de conversa do cliente. Pior: Quando os contêineres caem ou a conversa "esfria" de um dia para o outro, o LLM esquece que taxa propôs, perde o contexto, e irrita os clientes *Premium*.

## Seu Objetivo
Sua tarefa é finalizar o orquestrador `StatefulFinanceAgent` em `src/agent_router.py`.

### Passo a Passo:
1. Abra `src/agent_router.py`.
2. Localize a marcação `TODO (LAB-DESAFIO)`.
3. Adicione a lógica da Máquina de Estados (FSM). Você deve **incrementar** a recusa do cliente caso ele demonstre insatisfação, chamando o método `state.increment_rejection()`. (Para simplificar, conte todas as chamadas como incremento, já que o main manda mensagens negativas simuladas).
4. Em seguida, persista obrigatoriamente a sessão no Storage ADK usando: `self.session_gw.save_checkpoint(adk_session, state)`.

### Solução Esperada
A lógica que faltava é simples, mas demonstra como injetar o controle imperativo entre os turnos LLM.
```python
# =====================================================================
# TODO (LAB-DESAFIO): ATUALIZAR O ESTADO E SALVAR CHECKPOINT
# =====================================================================
state.increment_rejection()
if state.funnel_stage == "human_handoff":
    response.text += "\n\n[SYSTEM] Aviso: Limite de recusas excedido. O Circuit Breaker efetuará Handoff Humano."
    
self.session_gw.save_checkpoint(adk_session, state)
```

## Critérios de Sucesso
- Ao executar `python -m src.main`, na terceira mensagem do cliente, o fluxo deve interromper a geração com o aviso `Handoff acionado: Transferindo o cliente para analista humano.`
- A tabela final `FinOps: Custo de Memória` deve exibir a diferença entre a abordagem amnésica ($$$) e a sua abordagem FSM com Checkpointing.
