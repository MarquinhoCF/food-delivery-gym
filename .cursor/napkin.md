# Napkin Runbook

## Curation Rules
- Re-prioritize on every read.
- Keep recurring, high-value notes only.
- Max 10 items per category.
- Each item includes date + "Do instead".

## Execution & Validation (Highest Priority)
1. **[2026-10-09] Pytest vive no venv local**
   Do instead: `./venv/bin/python -m pytest …` (não o `python` do pyenv global).

## Domain Behavior Guardrails
1. **[2026-10-09] MCTS: ordenação míope isolada em `_order_untried_actions`**
   Do instead: trocar proxy de expansão (C(S,x), distância, etc.) só nesse método; reaproveitar `pending` na expansão.

2. **[2026-10-09] Novo hiperparâmetro MCTS: fio completo (catálogo/scripts/docs)**
   Do instead: espelhar `max_outcomes` — `MCTSVariantSpec`, `parse_mcts_cli`, `mcts_result_key` (sufixo opcional `_dthrK` se finito), builder, extras, experiment YAML, helps e docs.

3. **[2026-10-09] Proibido alterar testes existentes**
   Do instead: corrigir produção; testes novos são ok; pedir aprovação explícita antes de editar um teste antigo.

## User Directives
1. **[2026-10-09] Comentários/estilo MCTS em português**
   Do instead: manter comentários e docstrings do arquivo MCTS em português ao alterar.
