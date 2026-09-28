# Script `convert_metrics`

Converte arquivos de métricas entre NPZ (compacto) e JSON (legível por humanos).

```bash
python -m scripts.convert_metrics PATH [--name metrics_data] [--dry-run]
```

## Opções

| Opção | Descrição | Padrão |
|-------|-----------|--------|
| `path` | Arquivo `.npz`/`.json` ou diretório | obrigatório |
| `--name` | Nome-base em modo diretório | `metrics_data` |
| `--dry-run` | Lista conversões sem gravar arquivos | (ausente) |

## Exemplos

```bash
python -m scripts.convert_metrics data/runs/execucoes/obj_3/simple_scenario/random/metrics_data.npz

python -m scripts.convert_metrics data/runs/experimento_exemplo --dry-run

python -m scripts.convert_metrics data/runs/experimento_exemplo
```

Em modo diretório, a conversão é recursiva e grava o arquivo convertido ao lado do original.
