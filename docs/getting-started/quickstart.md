# Início rápido

Com o ambiente ativado e as dependências instaladas ([Instalação](installation.md)), valide o simulador com os comandos abaixo.

## 1. Modo interativo (recomendado para desenvolvimento)

```bash
python -m scripts.test_runner --mode interactive --scenario medium.json --render
```

Durante a execução, use estes comandos no terminal:

| Entrada | Efeito |
|---------|--------|
| `Enter` | Ação aleatória |
| `run` | Executa automaticamente até o fim |
| `quit` | Encerra |
| número | Seleciona o índice do motorista |

## 2. Modo automático

```bash
python -m scripts.test_runner --mode auto --scenario medium.json --optimizer random
```

## 3. Heurística `lowest` com função de custo

```bash
python -m scripts.test_runner \
  --mode auto \
  --scenario medium.json \
  --optimizer lowest \
  --lowest cost=route \
  --objective 3
```

## 4. Experimento em lote (exemplo versionado)

```bash
python -m scripts.run_batch_eval experiments/example.yaml --num-runs 2 --no-rl \
  --agents random nearest_driver --lowest cost=route
```

A saída fica em `data/runs/experimento_exemplo/` (conforme o campo `name` do YAML).

## 5. Relatório a partir do `run.json`

```bash
python -m scripts.report data/runs/experimento_exemplo
```

## Próximos passos

- [test_runner](../tools/test-runner.md): opções completas
- [Cenários](../framework/scenarios.md): como configurar experimentos
- [YAML de experimento](../experiments/yaml.md): avaliação em lote
- [RL Baselines3 Zoo](../rl-baselines3-zoo/README.md): treinamento PPO
