# Layout de dados

Os dados experimentais **não** são versionados no Git (`data/` está no `.gitignore`). Eles permanecem locais ou no [Google Drive do artigo](https://drive.google.com/drive/folders/1LPtYEpgLncWMga_ysa0-UeE2ng3i0UxC?usp=sharing).

## Experimentos nomeados (layout atual)

```
data/runs/<name>/
  run.json
  summary.csv
  report/                          # gerado por scripts.report
    objective_table.xlsx
    boxplots/
  obj_<N>/<scenario>/<agent>/
    episodes.csv
    metrics_data.npz               # ou metrics_data.json
    summary.json
    figs/                          # opcional
```

Exemplo: `data/runs/experimento_exemplo/obj_3/medium/nearest_driver/`.

## Layout legado (dados do artigo / tag `v1.2.8`)

Os dados do [Google Drive do SBPO](https://drive.google.com/drive/folders/1LPtYEpgLncWMga_ysa0-UeE2ng3i0UxC?usp=sharing) foram gerados na tag **`v1.2.8`**, neste formato:

```
data/runs/execucoes/obj_<N>/<scenario>_scenario/<agent>/
  results.txt
  metrics_data.npz
  figs/                 # opcional
```

Diferenças em relação ao layout novo:

| Aspecto | Novo | Legado (`v1.2.8` / artigo) |
|---------|------|----------------------------|
| Raiz | `data/runs/<name>/` | `data/runs/execucoes/` |
| Pasta do cenário | `<scenario>` | `<scenario>_scenario` |
| Manifesto | `run.json` | (ausente) |
| Resumo por episódio | `episodes.csv`, `summary.json` | `results.txt` |

Os scripts `generate_table`, `generate_boxplots`, `generate_plots` e `convert_metrics` ainda aceitam o layout legado (os defaults apontam para `./data/runs/execucoes`). O script `report` **não** se aplica a esse pacote, pois exige `run.json`.

Roteiro completo para revisores: [Reprodução SBPO](reproducing-sbpo.md).

## Modelos PPO

```
data/ppo_training/<cenário>/
  otimizacao_1M_steps_200_trials/obj_N/.../optimization/trial_K/
  treinamento/obj_N/<nome_do_modelo>/
    best_model.zip
    FoodDelivery-<cenário>-objN-v1/vecnormalize.pkl
    evaluations.npz
    *.monitor.csv
```

O nome da pasta `<nome_do_modelo>` vira o identificador do agente (por exemplo, `18M_steps` pode aparecer como `ppo_18M_steps` ou conforme o catálogo descobre).

## Figuras e tabelas do artigo (Drive)

No pacote de dados do artigo, além das pastas acima, podem existir:

- `data/figuras/`: boxplots finais
- `data/tabelas/objective_table.xlsx`: planilha consolidada

Os defaults atuais dos scripts gravam em `data/runs/figuras` e `data/runs/tabelas`, ou em `data/runs/<name>/report/` via `report`.

## Formatos de métricas

| Formato | Uso |
|--------|-----|
| `metrics_data.npz` | Padrão; compacto |
| `metrics_data.json` | Legível; útil para inspeção |

Conversão: [convert_metrics](../tools/convert-metrics.md).

## Documentação relacionada

- [Reprodução SBPO](reproducing-sbpo.md)
- [Artefatos do Zoo](../rl-baselines3-zoo/artifacts.md)
- [Convenções](../reference/conventions.md)
