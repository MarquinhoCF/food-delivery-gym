# Script `report`

Gera planilha e boxplots a partir de um experimento que possui `run.json`. Objetivos e cenários vêm da especificação; os agentes são descobertos no disco.

```bash
python -m scripts.report EXPERIMENT_DIR [--episodes]
```

## O que faz

- Cria `report/objective_table.xlsx` (mesma lógica de `generate_table`)
- Gera boxplots em `report/boxplots/` com prefixo `boxplot_obj<N>_`
- Com `--episodes`, também gera gráficos por episódio em cada pasta de agente

Sem `run.json`, o script encerra com mensagem de erro clara. Para pastas legadas, use [generate_table](generate-table.md) e [generate_boxplots](generate-boxplots.md) diretamente.

## Exemplos

```bash
python -m scripts.report data/runs/experimento_exemplo
python -m scripts.report data/runs/experimento_exemplo --episodes
```

## Layout de saída

```
data/runs/<name>/
  report/
    objective_table.xlsx
    boxplots/
      boxplot_obj3_delivery_time_scenarios.png
      boxplot_obj3_rewards_scenarios.png
      boxplot_obj3_distance_scenarios.png
  obj_3/<scenario>/<agent>/figs/    # apenas com --episodes
```
