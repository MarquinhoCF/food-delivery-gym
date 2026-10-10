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
1. **[2026-10-09] MCTS: ordem de expansão em `_order_untried_actions`**
   Do instead: `expansion_order=immediate` (C(S,x)+`pending`) ou `heuristic` (`ranked_actions` da base, expand sob demanda). Trocar proxy só nesse método.

2. **[2026-10-09] Novo hiperparâmetro MCTS: fio completo (catálogo/scripts/docs)**
   Do instead: espelhar `max_expanded_actions` / `expansion_order` — `MCTSVariantSpec`, `parse_mcts_cli`, `mcts_result_key` (sufixos `_maxexpK` / `_ordheur` só quando não-default), builder, extras, experiment YAML, helps e docs.

3. **[2026-10-09] `ranked_actions` só em bases com critério**
   Do instead: implementar em `nearest`/`lowest`/`random`; `first` não tem; `expansion_order=heuristic` + `first` deve falhar na CLI e no construtor.

4. **[2026-10-09] Proibido alterar testes existentes**
   Do instead: corrigir produção; testes novos são ok; pedir aprovação explícita antes de editar um teste antigo.

## User Directives
1. **[2026-10-09] Sem jargão do Powell em nomes públicos**
   Do instead: parâmetro `max_expanded_actions`; abreviação `maxexp` em chave de pasta e título. Não usar `d_thr`.

2. **[2026-10-09] Comentários/estilo MCTS em português**
   Do instead: manter comentários e docstrings do arquivo MCTS em português ao alterar.
