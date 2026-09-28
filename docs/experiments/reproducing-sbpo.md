# Reprodução dos resultados do SBPO 2026

Este roteiro destina-se a revisores e leitores que desejam inspecionar ou regenerar as figuras e tabelas do artigo **sem** reexecutar o treinamento completo.

## Versão dos dados

Os artefatos publicados no Google Drive foram gerados na tag **[`v1.2.8`](https://github.com/MarquinhoCF/food-delivery-gym/releases/tag/v1.2.8)**, com o pipeline e o layout **legados** da época.

Desde então, o repositório evoluiu. A tabela abaixo resume as diferenças mais relevantes.

| Aspecto | Tag `v1.2.8` (dados do artigo) | Código atual (HEAD) |
|---------|--------------------------------|---------------------|
| Pasta de avaliações | `data/runs/execucoes/obj_N/<cenário>_scenario/<agente>/` | `data/runs/<nome>/obj_N/<cenário>/<agente>/` |
| Manifesto | (ausente) | `run.json` + `summary.csv` |
| Resumo por episódio | `results.txt` | `episodes.csv` + `summary.json` |
| Métricas | `metrics_data.npz` (ou `.json`) | idem |
| Definição do experimento | apenas flags na CLI | YAML + CLI ([`experiments/example.yaml`](../../experiments/example.yaml)) |
| Relatório unificado | `scripts.report` inexistente | `python -m scripts.report` |
| Heurísticas no batch | nomes fixos (`lowest_route_cost`, …) | catálogo com `lowest` / `rollout` e variantes |
| Default de episódios no batch | `20` | `20` (o artigo usou **30**) |

Para **inspecionar ou regenerar figuras** a partir do Drive, use o layout legado e os scripts descritos neste documento. Para **reexecutar as simulações exatamente como no artigo**, faça checkout da tag `v1.2.8` e rode o batch com `--num-runs 30`.

## Dados disponíveis

- **Link:** [Dados do Artigo (Google Drive)](https://drive.google.com/drive/folders/1LPtYEpgLncWMga_ysa0-UeE2ng3i0UxC?usp=sharing)
- O acesso é concedido **mediante solicitação** pelo Google Drive.
- Os dados brutos usados nas figuras e tabelas estão nesse diretório. Não é obrigatório re-simular para conferir os resultados.

Metadados da publicação: consulte o [README principal](../../README.md#publicação-no-sbpo-2026).

## Escopo do artigo

| Item | Valor |
|------|--------|
| Tag do código usada na geração | **`v1.2.8`** |
| Objetivo de recompensa | **3** (tempo efetivo gasto pelos motoristas) |
| Cenários | `simple`, `medium`, `complex` |
| Episódios por agente na avaliação final | **30** |
| Default do `run_batch_eval` naquela tag | `20` (o artigo sobrescreveu com `--num-runs 30`) |

## Organização dos dados no Drive e na pasta `data/`

Organize a pasta baixada de forma que a raiz do repositório enxergue `data/` conforme as seções abaixo.

### Treinamento PPO (`data/ppo_training/<cenário>/`)

Para cada cenário (`simple`, `medium`, `complex`):

| Pasta | Conteúdo |
|-------|----------|
| `otimizacao_1M_steps_200_trials/obj_3/...` | 200 trials Optuna (`best_model.zip`, `evaluations.npz`, relatórios) |
| `treinamento/obj_3/18M_steps/` | PPO padrão, 18M passos |
| `treinamento/obj_3/18M_steps_otimizado/` | PPO com hiperparâmetros do melhor trial, 18M passos |

Cada treino final inclui tipicamente `best_model.zip`, `evaluations.npz`, monitores (`*.monitor.csv`) e `vecnormalize.pkl`.

### Avaliações finais (layout `v1.2.8`)

```
data/runs/execucoes/
  obj_3/
    simple_scenario/<agente>/
    medium_scenario/<agente>/
    complex_scenario/<agente>/
```

Agentes tipicamente presentes no artigo:

- Heurísticas: `random`, `first_driver`, `nearest_driver`, `lowest_route_cost`, `lowest_marginal_route_cost`
- PPO: `ppo_18M_steps`, `ppo_18M_steps_otimizado`

Arquivos por agente:

| Arquivo | Conteúdo |
|---------|----------|
| `results.txt` | Log textual das 30 execuções (recompensa, tempo SimPy, truncamento) e resumo estatístico |
| `metrics_data.npz` | Dados estruturados (séries por episódio, motoristas e estabelecimentos, eventos) |
| `figs/` | Gráficos já renderizados (episódio e/ou lote), quando gerados com `--batch-plots`, `--all-plots` ou `generate_plots` |

Figuras e tabela finais do artigo também podem estar em:

- `data/figuras/boxplot_*_scenarios.png`
- `data/tabelas/objective_table.xlsx`

## Scripts da tag `v1.2.8` (pipeline do artigo)

A organização dos dados e o funcionamento completo dos scripts usados na geração dos resultados do artigo estão documentados no [`README.md` da tag `v1.2.8`](https://github.com/MarquinhoCF/food-delivery-gym/blob/v1.2.8/README.md). Consulte esse arquivo para a referência oficial do pipeline legado (opções de CLI, exemplos e estrutura de saída).

Naquela versão, o pós-processamento e a avaliação em lote giravam em torno dos scripts abaixo (não havia `scripts.report` nem YAML de experimento):

| Script | Papel |
|--------|--------|
| `scripts.run_batch_eval` | Avaliações em lote; saída em `data/runs/execucoes/obj_{}/{}_scenario/` |
| `scripts.generate_plots` | Gráficos por episódio e agregados a partir de `metrics_data.*` |
| `scripts.generate_table` | Planilha Excel consolidada |
| `scripts.convert_metrics` | Conversão NPZ ↔ JSON |
| `scripts.generate_boxplots` | Boxplots comparativos |

Características do `run_batch_eval` naquela versão:

- apenas CLI;
- `--results-base-dir` com placeholders, padrão `./data/runs/execucoes/obj_{}/{}_scenario/`;
- heurísticas com nomes fixos (`--heuristics random nearest_driver ...`);
- modelos descobertos em `data/ppo_training/<cenário>/treinamento/obj_N/<nome>/best_model.zip`.

Exemplo alinhado ao artigo (na tag `v1.2.8`):

```bash
git checkout v1.2.8

python -m scripts.run_batch_eval \
  --objectives 3 \
  --scenarios simple medium complex \
  --num-runs 30 \
  --experiment-mode same_scenario
```

## Roteiro

Os scripts atuais `generate_*` e `convert_metrics` ainda aceitam o layout legado `execucoes` / `<cenário>_scenario`. 

**Não** use `scripts.report` sobre os dados do Drive, pois ele exige `run.json`.

1. Baixe os dados e coloque a pasta `data/` na raiz do repositório.
2. Inspecione ou converta métricas:

```bash
python -m scripts.convert_metrics data/runs/execucoes --dry-run
python -m scripts.convert_metrics \
  data/runs/execucoes/obj_3/simple_scenario/random/metrics_data.npz
```

3. Regenere a planilha:

```bash
python -m scripts.generate_table \
  -r data/runs/execucoes \
  -out data/tabelas/objective_table.xlsx \
  -o 3 \
  -s simple medium complex
```

4. Regenere os boxplots:

```bash
python -m scripts.generate_boxplots \
  -r data/runs/execucoes \
  -o 3 \
  -s simple medium complex \
  -od data/figuras
```

5. Regenere gráficos por agente (opcional):

```bash
python -m scripts.generate_plots -r data/runs/execucoes -o 3
```

## Reexecutar o pipeline completo

| Objetivo | Como |
|----------|------|
| Regenerar só figuras e tabela a partir do Drive | Código atual + comandos da seção anterior |
| Reexecutar avaliações iguais às do artigo | Checkout `v1.2.8` + `run_batch_eval` com `--num-runs 30` |
| Entender o treino PPO e Optuna | [RL Baselines3 Zoo](../rl-baselines3-zoo/README.md) (fluxo ainda baseado no fork externo) |

## Ambiente

Para regenerar figuras no código atual, siga [Instalação](../getting-started/installation.md). Para aproximar o ambiente de desenvolvimento da época, use **Python 3.10.12** e [`requirements-freeze.txt`](../../requirements-freeze.txt).

Se for reexecutar simulações na tag `v1.2.8`, faça checkout dessa tag e instale as dependências **dela** (freeze e scripts correspondentes).
