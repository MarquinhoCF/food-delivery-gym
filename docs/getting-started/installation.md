# Instalação

Este guia descreve como preparar o ambiente para usar o Food Delivery Gym.

## Requisitos

| Item | Valor |
|------|--------|
| Python (suporte formal do pacote) | `>= 3.10` ([`pyproject.toml`](../../pyproject.toml)) |
| Python (reprodução exata do ambiente de desenvolvimento) | **3.10.12** |
| Sistema | Linux recomendado (especialmente para GPU/CUDA) |
| Dependências do pacote mínimo | SimPy, Gymnasium, NumPy, Matplotlib, Pygame, PyYAML |

Há três formas de instalar:

1. **Instalação padrão:** `requirements.txt` (flexível)
2. **Instalação editável:** desenvolvimento local
3. **Reprodução exata:** `requirements-freeze.txt` + Python **3.10.12**

---

## 1. Ambiente virtual

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
.\venv\Scripts\activate
```

Com **pyenv**, para criar o ambiente com Python **3.10.12** (reprodução exata):

```bash
pyenv install 3.10.12   # se ainda não estiver instalada
"$(pyenv root)/versions/3.10.12/bin/python" -m venv venv
source venv/bin/activate
python --version        # deve mostrar 3.10.12
```

---

## 2. Instalação padrão (`requirements.txt`)

Esta opção é adequada para uso geral do simulador, scripts e dependências de RL:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

O arquivo [`requirements.txt`](../../requirements.txt) inclui, entre outras, Stable-Baselines3, `rl_zoo3`, OR-Tools, TensorFlow/TensorBoard e utilitários de análise.

A instalação editável (`pip install -e .`) registra o pacote `food_delivery_gym` e os ambientes Gymnasium no import.

---

## 3. Reprodução exata (`requirements-freeze.txt`)

Use este caminho para mimetizar o ambiente da máquina de desenvolvimento (versões fixadas):

1. Use **Python 3.10.12**.
2. Instale a partir de [`requirements-freeze.txt`](../../requirements-freeze.txt) (nome com **hífen**, não underscore).

```bash
# Confirme a versão
python --version   # Python 3.10.12

python -m pip install --upgrade pip
python -m pip install -r requirements-freeze.txt
python -m pip install -e .
```

### Observações importantes

- O freeze foi gerado em **Linux com CUDA 12.x** e inclui pacotes `nvidia-*`, `torch` e `triton`.
- Em **Windows**, **macOS** ou ambiente **somente CPU**, algumas entradas podem falhar. Nesse caso:
  - prefira a instalação padrão com `requirements.txt`; ou
  - instale o freeze ignorando pacotes incompatíveis com a plataforma e complete PyTorch pelo [seletor oficial](https://pytorch.org/get-started/locally/).
- O suporte formal do pacote continua sendo `Python >= 3.10`. A pinagem **3.10.12** aplica-se especificamente à reprodução via freeze.

Versões relevantes pinadas no freeze (referência):

| Pacote | Versão no freeze |
|--------|------------------|
| `gymnasium` | 1.2.1 |
| `stable_baselines3` | 2.7.0 |
| `sb3_contrib` | 2.7.0 |
| `rl_zoo3` | 2.7.0 |
| `torch` | 2.8.0 |

---

## 4. Dependências do sistema (Linux)

Para gráficos Matplotlib com backend Tk:

```bash
sudo apt-get install python3-tk
```

Para renderização com Pygame, garanta suporte gráfico no sistema (display disponível).

---

## 5. Arquivo `.env`

O `test_runner` carrega variáveis de ambiente via `python-dotenv`:

```bash
cp .env.example .env
```

Conteúdo padrão:

```
WINDOW_WIDTH=1600
WINDOW_HEIGHT=1000
DRAW_GRID=True
FPS=30
```

Sem `.env`, a execução com renderização pode falhar ao ler essas variáveis.

---

## 6. Verificação da instalação

```bash
python -c "import food_delivery_gym; import gymnasium; print('ok')"
python -m scripts.test_runner --mode auto --scenario simple.json --max-steps 50
pytest
```

---

## 7. Treinamento com RL Baselines3 Zoo

O treino oficial de agentes PPO **não** fica confinado a este repositório. Ele usa o fork [MarquinhoCF/rl-baselines3-zoo](https://github.com/MarquinhoCF/rl-baselines3-zoo), baseado no [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo). Este projeto fornece o ambiente Gymnasium; o Zoo fornece `train.py`, Optuna e os utilitários de hiperparâmetros.

O layout recomendado é colocar os dois repositórios lado a lado e usar **dois ambientes virtuais separados**:

```
projetos/
  food-delivery-gym/
    venv/                 # simulação, avaliação, scripts
  rl-baselines3-zoo/
    venv/                 # treino PPO e Optuna
```

| Ambiente | Onde criar | Uso |
|----------|------------|-----|
| Venv do gym | `food-delivery-gym/` (seções 1 a 6) | `test_runner`, `run_batch_eval`, `report`, etc. |
| Venv do Zoo | `rl-baselines3-zoo/` | `train.py`, Optuna, `update_ppo_envs.py` |

No venv do Zoo ainda é preciso instalar o `food-delivery-gym` em modo editável (`pip install -e .`), para que os IDs `FoodDelivery-*` fiquem registrados no Gymnasium durante o treino. Isso **não** substitui o venv do gym: são instalações independentes.

Após a instalação, o fluxo de uso (Optuna, treino, artefatos) está em [RL Baselines3 Zoo](../rl-baselines3-zoo/README.md).

### Passo a passo (venv do Zoo)

Antes de seguir, conclua a instalação do `food-delivery-gym` com o **venv dele** (seções 1 a 6).

**1º passo.** Clone o fork do RL Baselines3 Zoo (idealmente ao lado do `food-delivery-gym`):

```bash
git clone https://github.com/MarquinhoCF/rl-baselines3-zoo.git
```

**2º passo.** Entre na raiz do `rl-baselines3-zoo`:

```bash
cd rl-baselines3-zoo
```

**3º passo.** Crie e ative um **novo** ambiente virtual Python (separado do venv do gym). Para reprodução exata, use **Python 3.10.12** (veja a seção 1).

```bash
python3 -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
```

**4º passo.** Instale as dependências do Zoo:

```bash
python -m pip install -r requirements.txt
```

**5º passo.** Instale as dependências adicionais usadas no fluxo de treino e análise:

```bash
pip install huggingface_hub huggingface-sb3 sb3-contrib optuna rl_zoo3 seaborn scipy
```

**6º passo.** Navegue até o diretório do `food-delivery-gym` (mantenha o **venv do Zoo** ativo):

```bash
cd ../food-delivery-gym/
```

**7º passo.** Instale o pacote local em modo editável **neste venv do Zoo** (isso registra os ambientes `FoodDelivery-*` para o `train.py`):

```bash
pip install -e .
```

Se quiser mimetizar versões pinadas também no venv de treino (Linux com CUDA 12.x), use o freeze **antes** do `-e .`:

```bash
pip install -r requirements-freeze.txt
pip install -e .
```

O arquivo [`requirements-freeze.txt`](../../requirements-freeze.txt) fixa as versões usadas no desenvolvimento. Em Windows, macOS ou ambiente somente CPU, algumas entradas (`nvidia-*`, `triton`, etc.) podem falhar; nesse caso, instale o que for compatível com a plataforma e complete o PyTorch pelo [seletor oficial](https://pytorch.org/get-started/locally/).

**8º passo.** Volte para o diretório do `rl-baselines3-zoo` (os comandos de treino e Optuna rodam a partir daí, com o venv do Zoo ativo):

```bash
cd ../rl-baselines3-zoo/
```

### Verificação

Com o **venv do Zoo** ativo e estando na raiz do Zoo:

```bash
python -c "import food_delivery_gym; import gymnasium as gym; print(gym.spec('FoodDelivery-medium-obj3-v1'))"
```

Se você adicionou cenários ou objetivos novos, atualize o `hyperparams/ppo.yml` do Zoo:

```bash
python scripts/update_ppo_envs.py
```

Para avaliação no simulador (`run_batch_eval`, `report`, etc.), ative o **venv do gym** e trabalhe a partir de `food-delivery-gym/`.

### Problemas frequentes na configuração

Consulte [Problemas comuns](../rl-baselines3-zoo/troubleshooting.md) para detalhes. Em resumo:

| Sintoma | Ação típica |
|---------|-------------|
| `No module named 'pkg_resources'` | `pip install "setuptools<81"` |
| Conflito entre SB3, `sb3_contrib` e `rl_zoo3` | Alinhe às versões do [`requirements-freeze.txt`](../../requirements-freeze.txt) |
| Erros de API do Gymnasium | Evite misturar `gym` (legado) e `gymnasium`; use a versão do freeze |
| CUDA / GPU indisponível | Verifique `torch.cuda.is_available()` e reinstale o PyTorch para o seu driver |
| `ModuleNotFoundError: food_delivery_gym` no Zoo | Com o venv do Zoo ativo, rode `pip install -e .` em `food-delivery-gym/` |

Após a configuração, os próximos passos são o [ajuste de hiperparâmetros](../rl-baselines3-zoo/hyperparameter-tuning.md) e o [treinamento](../rl-baselines3-zoo/training.md).

---

## Pacote mínimo vs. stack completa

| Forma | O que instala | Quando usar |
|-------|---------------|-------------|
| `pip install .` / `-e .` sem requirements | Apenas dependências do [`pyproject.toml`](../../pyproject.toml) | Uso mínimo do ambiente Gymnasium |
| `requirements.txt` | Stack completa (RL, OR-Tools, plots, etc.) | Uso diário recomendado |
| `requirements-freeze.txt` | Versões exatas do desenvolvimento | Reprodução e depuração de compatibilidade |

---

## Próximos passos

- [Início rápido](quickstart.md)
- [Visão geral do framework](../framework/overview.md)
- [RL Baselines3 Zoo](../rl-baselines3-zoo/README.md)
