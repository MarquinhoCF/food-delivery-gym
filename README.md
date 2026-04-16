# Food Delivery Simulator

## 📋 Visão Geral

O **Food Delivery Simulator** é um simulador de entrega de comida desenvolvido utilizando a biblioteca de simulação de eventos discretos **SimPy**. O simulador foi adaptado para funcionar como um ambiente compatível com **Gymnasium**, permitindo experimentos com **Aprendizado por Reforço**. O objetivo é testar e treinar agentes utilizando algoritmos como **PPO (Proximal Policy Optimization)** da biblioteca **Stable-Baselines3**.

![Simulador de Delivery de Comida](simulator.gif)

## 📋 Requisitos

- Python 3.10 ou superior
- Gymnasium
- SimPy
- Stable-Baselines3
- RL Baselines3 Zoo
- Outras dependências listadas em `requirements.txt`

## ⚙️ Configuração do Ambiente

### 1️⃣ Criar e ativar o ambiente virtual do Python

#### No Windows:
```shell
python -m venv venv
Set-ExecutionPolicy Unrestricted -Scope Process
.\venv\Scripts\activate
```

#### No Linux/Mac:
```shell
python3 -m venv venv
source venv/bin/activate
```

### 2️⃣ Instalar as dependências
```shell
python -m pip install -r requirements.txt
```

### 3️⃣ (Apenas no Linux) Instalar `python-tk` para usar o Matplotlib
```shell
sudo apt-get install python3-tk
```

### 4️⃣ Criar uma copia do exemplo de arquivo de ambiente
```shell
cp .env.example .env
```

## 🚀 Uso do Simulador

### 🔹 Sempre que for usar o script Python, ative o ambiente virtual:

#### No Windows:
```shell
Set-ExecutionPolicy Unrestricted -Scope Process
.\venv\Scripts\activate
```

#### No Linux/Mac:
```shell
source venv/bin/activate
```

### 🔹 Rodar o script do teste do simulador:

#### 1. **Modo Automático**
```shell
python -m scripts.test_runner --mode auto --scenario medium.json --render
```
- Executa automaticamente com ações aleatórias

#### 2. **Modo Interativo** (Recomendado para desenvolvimento)
```shell
python -m scripts.test_runner --mode interactive --scenario medium.json --render
```
- Executa passo-a-passo esperando sua entrada
- **Comandos disponíveis**:
  - `Enter`: ação aleatória
  - `run`: executa automaticamente até o fim
  - `quit`: sai do programa
  - Números/ações: entrada manual personalizada

#### 3. **Modo com Agente PPO** (Requer modelo treinado)
```shell
python -m scripts.test_runner --mode auto --optimizer rl --model-path models/ppo_food_delivery/best_model.zip --render
```
- Usa `--optimizer rl` combinado com `--mode auto`
- O `vecnormalize.pkl` é carregado automaticamente se encontrado no mesmo diretório do modelo

### ⚙️ Opções de Configuração

| Opção | Descrição | Exemplo |
|-------|-----------|---------|
| `--mode` | Modo de execução: `interactive`, `auto` | `--mode interactive` |
| `--scenario` | Arquivo de cenário JSON | `--scenario medium.json` |
| `--optimizer` | Otimizador: `random`, `first`, `nearest`, `lowest`, `rl` | `--optimizer lowest` |
| `--objective` | Objetivo de recompensa do ambiente (1-13) | `--objective 1` |
| `--cost-function` | Função de custo usada pelo `lowest`: `route`, `marginal_route` | `--cost-function route` |
| `--model-path` | Caminho para o `best_model.zip` do PPO (obrigatório com `--optimizer rl`) | `--model-path models/ppo_model/best_model.zip` |
| `--render` | Ativa visualização gráfica | `--render` |
| `--seed` | Seed para reproducibilidade | `--seed 42` |
| `--max-steps` | Limite máximo de passos | `--max-steps 1000` |
| `--save-log` | Salva output em arquivo log.txt | `--save-log` |

> A opção `--cost-function` é obrigatória quando `--optimizer lowest` está selecionado, e inválida nos demais casos.
> A opção `--model-path` é obrigatória quando `--optimizer rl` está selecionado.

## 🎯 Configuração dos Cenários Experimentais

O ambiente de simulação foi formulado como um **Processo de Decisão de Markov (MDP)** voltado ao **problema de entrega de última milha**.
A seguir são apresentadas as principais constantes e parâmetros utilizados na criação dos **cenários experimentais**.

### 📊 Parâmetros Principais

| Variável             | Descrição                                                             |
| -------------------- | --------------------------------------------------------------------- |
| `num_drivers`        | Número total de motoristas disponíveis.                               |
| `num_orders`         | Total de pedidos a serem gerados na simulação.                        |
| `num_establishments` | Quantidade de restaurantes ou estabelecimentos.                       |
| `grid_map_size`      | Tamanho do mapa da cidade (em um grid quadrado, por exemplo `50x50`). |
| `max_time_step`      | Tempo máximo da simulação (em minutos).                               |

---

### 📦 Geração de Pedidos (Processos de Poisson)

A criação de novos pedidos é controlada por um gerador configurável no campo `order_generator`.
Atualmente, o ambiente suporta dois tipos de geradores:

| Tipo                        | Classe Interna                           | Descrição                                                                     |
| --------------------------- | ---------------------------------------- | ----------------------------------------------------------------------------- |
| `"poisson"`                 | `PoissonOrderGenerator`                  | Gera pedidos de forma homogênea com taxa constante λ.                         |
| `"non_homogeneous_poisson"` | `NonHomogeneousPoissonOrderGenerator`    | Gera pedidos de forma não homogênea com taxa variável no tempo (função λ(t)). |

#### 🔹 Parâmetros Disponíveis

| Campo                   | Obrigatório | Descrição                                                                                                |
| ----------------------- | ----------- | -------------------------------------------------------------------------------------------------------- |
| `type`                  | ✅          | Define o tipo de processo: `"poisson"` ou `"non_homogeneous_poisson"`.                                   |
| `estimated_num_orders`  | ✅          | Número inteiro positivo com o total estimado de pedidos a serem gerados.                                 |
| `time_window`           | ✅          | Janela total de tempo da geração de pedidos (em minutos). Deve ser um valor positivo.                    |
| `lambda_rate`           | ❌          | Taxa média de chegada (λ) — usada apenas no gerador `"poisson"`. Se omitida, o sistema calcula como `estimated_num_orders / time_window`. |
| `rate_function`         | ✅ (não homogêneo) | Função lambda que define a taxa variável de chegada λ(t) — obrigatória no gerador `"non_homogeneous_poisson"`. |
| `max_rate`              | ❌          | Taxa máxima usada para o método de *thinning* — usada no gerador `"non_homogeneous_poisson"`. Se omitida, é calculada automaticamente. |

#### 🔸 Exemplo — Processo de Poisson Homogêneo

```json
"order_generator": {
  "type": "poisson",
  "estimated_num_orders": 288,
  "time_window": 1440,
  "lambda_rate": 0.2
}
```

Nesse exemplo, pedidos são gerados de acordo com um **processo de Poisson homogêneo** com taxa média de **0,2 pedidos por minuto** durante um período de **1440 minutos** (1 dia).

#### 🔸 Exemplo — Processo de Poisson Não Homogêneo

```json
"order_generator": {
    "type": "non_homogeneous_poisson",
    "estimated_num_orders": 576,
    "time_window": 960,
    "rate_function": "lambda t: 0.3115 + 0.9345 * (np.exp(-((t - 330)**2) / 7000) + np.exp(-((t - 630)**2) / 7000))"
}
```

Neste caso, a taxa de geração de pedidos **varia ao longo do tempo**, simulando períodos de alta e baixa demanda (por exemplo, picos no horário de almoço e jantar).

#### 🛠️ Poisson Tuner — Ferramenta Visual para Criação da Função de Taxa

Criar e calibrar a `rate_function` manualmente pode ser trabalhoso. Para facilitar esse processo, foi criado o **[Poisson Tuner](https://github.com/MarquinhoCF/poisson-tuner)** uma interface visual em React que permite configurar, visualizar e exportar funções de taxa λ(t) prontas para uso no simulador.

**O que ele oferece:**

- Configuração dos **parâmetros globais**: janela de tempo, número de pedidos desejados e taxa base
- Adição e remoção de **picos de demanda** com controle individual de centro (minuto do pico), intensidade e largura (dispersão do pico)
- **Ajuste automático de escala**: a ferramenta calcula um fator de escala e ajusta todos os parâmetros para que o número esperado de pedidos bata exatamente com o valor desejado
- **Gráfico da taxa de chegada** λ(t) ao longo do tempo, comparando a função original e a ajustada
- **Gráfico de pedidos acumulados**, com linha de referência na meta definida
- **Importação de código**: cole uma `rate_function` existente e a ferramenta a parseia, carregando os parâmetros automaticamente nos controles
- **Exportação do código Python** pronto para copiar e colar diretamente no arquivo JSON do cenário

---

### 🚗 Configurações dos Motoristas

| Variável                | Obrigatório | Descrição                                                                                  |
| ----------------------- | ----------- | ------------------------------------------------------------------------------------------ |
| `num`                   | ✅          | Número inteiro positivo com o total de motoristas.                                         |
| `vel`                   | ✅          | Lista `[mínimo, máximo]` de velocidade dos motoristas. Exemplo: `[3, 5]`.                  |
| `tolerance_percentage`  | ✅          | Percentual (≥ 0) de tolerância de piora no tempo de entrega ao reordenar rotas.            |
| `max_capacity`          | ✅          | Número máximo inteiro positivo de pedidos simultâneos que um motorista pode carregar.       |

---

### 🏪 Configurações dos Estabelecimentos

| Variável                        | Obrigatório | Descrição                                                                                        |
| ------------------------------- | ----------- | ------------------------------------------------------------------------------------------------ |
| `num`                           | ✅          | Número inteiro positivo com o total de estabelecimentos.                                         |
| `prepare_time`                  | ✅          | Tempo de preparo dos pedidos `[mínimo, máximo]` (em minutos). Deve estar em ordem crescente.    |
| `operating_radius`              | ✅          | Raio de operação `[mínimo, máximo]` (em unidades do grid). Deve estar em ordem crescente.       |
| `production_capacity`           | ✅          | Capacidade de produção (número de cozinheiros) `[mínimo, máximo]`. Deve estar em ordem crescente.|
| `percentage_allocation_driver`  | ✅          | Fração de conclusão do preparo (entre `0` e `1`) para acionar a alocação do motorista.          |

---

### 🎛️ Alocação

O parâmetro `percentage_allocation_driver`, definido dentro da seção `establishments`, controla o percentual de preparo necessário para acionar a alocação do motorista — por exemplo, `0.7` significa que o motorista é chamado quando 70% do preparo estiver concluído.

---

### 💾 Exemplo de Cenário Experimental

O cenário é configurado por um arquivo JSON dentro de `food_delivery_gym/main/scenarios/`, como mostrado a seguir:

```json
{
    "order_generator": {
      "type": "poisson",
      "estimated_num_orders": 288,
      "time_window": 1440
    },
    "simpy_env": {
        "max_time_step": 2880
    },
    "grid_map": {
        "size": 50
    },
    "drivers" : {
        "num": 10,
        "vel": [3, 5],
        "tolerance_percentage": 50,
        "max_capacity": 2
    },
    "establishments": {
        "num": 10,
        "prepare_time": [20, 60],
        "operating_radius": [5, 30],
        "production_capacity": [4, 4],
        "percentage_allocation_driver": 0.7
    }
}
```

Esse cenário define:

* 10 motoristas e 10 estabelecimentos;
* 288 pedidos ao longo de 1440 minutos e tempo limite de 2880 minutos;
* geração de pedidos por **processo de Poisson homogêneo**.

### 📝 Registro do Cenário Experimental

O registro dos cenários é feito automaticamente no arquivo `food_delivery_gym/__init__.py`. Para cada combinação de cenário e objetivo de recompensa, um ambiente Gymnasium é registrado no formato:
```
FoodDelivery-{cenário}-obj{N}-v1
```

Os **cenários disponíveis** são descobertos automaticamente a partir dos arquivos `.json` presentes em `food_delivery_gym/main/scenarios/`.

Os **objetivos de recompensa disponíveis** são lidos diretamente de `FoodDeliveryGymEnv.REWARD_OBJECTIVES`. As descrições de cada objetivo estão no arquivo `food_delivery_gym/main/scenarios/reward_objectives.txt`.

Para registrar um novo cenário, basta adicionar o arquivo JSON correspondente em `food_delivery_gym/main/scenarios/` ele será detectado automaticamente. Para adicionar um novo objetivo de recompensa, atualize `REWARD_OBJECTIVES` em `FoodDeliveryGymEnv` e implemente a lógica correspondente no método `_calculate_reward`.

## 🤖 Treinamento de Agentes de Aprendizado por Reforço

Antes de iniciar o treinamento de agentes de Aprendizado por Reforço (AR), é necessário **definir um cenário experimental** a partir da seção anterior.

O processo de ajuste de hiperparâmetros e o treinamento será realizado utilizando a biblioteca [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo), que fornece uma interface robusta para experimentação com algoritmos como PPO, DQN, A2C, entre outros.

### 1️⃣ Configuração do RL Baselines3 Zoo

Siga os passos abaixo para preparar o ambiente de treinamento:

**1º Passo**: Clone o repositório forkado do RL Baselines3 Zoo:

```bash
git clone https://github.com/MarquinhoCF/rl-baselines3-zoo.git
```

**2º Passo**: Entre na raiz do projeto `rl-baselines3-zoo`:

```bash
cd rl-baselines3-zoo
```

**3º Passo**: Crie e ative um ambiente virtual Python:

```bash
python3 -m venv venv
source venv/bin/activate  # No Windows, use: venv\Scripts\activate
```

**4º Passo**: Instale as dependências do Zoo:

```bash
python -m pip install -r requirements.txt
```

**5º Passo**: Instale as dependências adicionais:

```bash
pip install huggingface_hub huggingface_sb3 sb3-contrib optuna rl_zoo3 seaborn scipy
```

**6º Passo**: Navegue até o diretório do projeto `food-delivery-gym`:

```bash
cd ../food-delivery-gym/
```

**7º Passo**: Instale o pacote local:

```bash
pip install -e .
```

**8º Passo**: Volte para o diretório do `rl-baselines3-zoo`:

```bash
cd ../rl-baselines3-zoo/
```

> 💡 Se quiser reproduzir o ambiente exato utilizado no desenvolvimento, o arquivo `requirements_freeze.txt` na raiz do repositório contém todas as versões fixadas. Para instalá-las diretamente:
> ```bash
> pip install -r requirements_freeze.txt
> ```
> Isso é especialmente útil ao depurar problemas de compatibilidade.
> - ⚠️ O requirements_freeze.txt contém versões compiladas para Linux com CUDA 12.x. Em Windows ou macOS, ou ao usar CPU, algumas entradas (como nvidia-* e triton) podem falhar — nesse caso, instale-as individualmente ignorando os pacotes incompatíveis com sua plataforma.

---

### 🛠️ Troubleshooting — Configuração do RL Baselines3 Zoo

#### ❌ `ModuleNotFoundError: No module named 'pkg_resources'`

Se pacote `pkg_resources` não foi instalado. Verifique se o ambiente está ativado e veja a versão do `setuptools`:

```bash
pip show setuptools
```

A partir das versões recentes do `setuptools` (81+), o módulo `pkg_resources` não é mais garantido / pode não vir incluído. Para resolver, instale uma versão anterior do `setuptools`:

```bash
pip install "setuptools<81"
```

#### ❌ `ImportError` ou comportamento inesperado envolvendo `stable_baselines3` ou `sb3_contrib`

Versões incompatíveis entre `stable_baselines3`, `sb3_contrib` e `rl_zoo3` são uma causa comum de erros silenciosos ou crashes. Certifique-se de usar as versões validadas em conjunto:

```bash
pip install stable_baselines3==2.7.0 sb3-contrib==2.7.1 rl_zoo3==2.7.0
```

#### ❌ Erros relacionados ao `gymnasium` (ex: `API` incompatível, `step()` retornando 4 valores em vez de 5)

O projeto requer `gymnasium==1.2.3`. A API mudou entre versões e wrappers antigos podem quebrar silenciosamente:

```bash
pip install gymnasium==1.2.3
```

Evite misturar `gym` (legado) e `gymnasium` no mesmo ambiente virtual.

#### ❌ Erros de CUDA / GPU (ex: `CUDA error`, `device-side assert`, `torch` não reconhece a GPU)

O projeto foi validado com `torch==2.10.0` e CUDA 12.x. Se estiver usando GPU, verifique a compatibilidade com o driver instalado:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Se `torch.cuda.is_available()` retornar `False`, reinstale o PyTorch com o índice correto para sua versão de CUDA. Consulte o seletor oficial em [pytorch.org/get-started](https://pytorch.org/get-started/locally/).

> Se não tiver GPU, o treinamento funciona normalmente em CPU — apenas mais lento.

---

### 2️⃣ Ajuste de Hiperparâmetros (Opcional, mas Recomendado)

> ⚠️ **Pré-requisito:** Caso tenha adicionado um novo cenário ou objetivo de recompensa ao `food_delivery_gym` desde a última vez que o `ppo.yml` foi atualizado, execute o script `update_ppo_envs.py` na raiz do `rl-baselines3-zoo` antes de prosseguir. Ele detecta automaticamente todos os ambientes registrados no pacote e atualiza a seção do `food_delivery_gym` no `hyperparams/ppo.yml`:
>
> ```bash
> python scripts/update_ppo_envs.py
> ```

O ajuste de hiperparâmetros usa o [Optuna](https://optuna.org/) para encontrar automaticamente a melhor configuração para o seu agente. Os resultados podem ser **salvos em um banco de dados SQLite**, permitindo que você **retome o estudo de onde parou** caso a execução seja interrompida.

#### 🔹 Comando básico com SQLite (recomendado)
 
```bash
python train.py \
  --algo ppo \
  --env FoodDelivery-medium-obj1-v1 \
  --n-timesteps 1000000 \
  --optimize-hyperparameters \
  --max-total-trials 200 \
  --n-jobs 2 \
  --storage sqlite:///optuna_studies.db \
  --study-name ppo_medium_obj1
```
 
#### 🔹 Retomando um estudo existente
 
Caso o processo seja interrompido, basta rodar o **mesmo comando novamente** com o mesmo `--storage` e `--study-name`. O Optuna detectará automaticamente o estudo salvo e continuará de onde parou, sem repetir trials já concluídos:
 
```bash
python train.py \
  --algo ppo \
  --env FoodDelivery-medium-obj1-v1 \
  --n-timesteps 1000000 \
  --optimize-hyperparameters \
  --max-total-trials 200 \
  --n-jobs 2 \
  --storage sqlite:///optuna_studies.db \
  --study-name ppo_medium_obj1
```
 
#### 📋 Parâmetros explicados
 
| Parâmetro | Descrição |
|-----------|-----------|
| `--algo ppo` | Algoritmo de AR a ser usado. Outros suportados: `a2c`, `dqn`, `sac`, `td3`. |
| `--env` | ID do ambiente Gymnasium registrado no pacote `food_delivery_gym`. |
| `--n-timesteps` | Número de passos de simulação por trial durante a otimização. Valores maiores são mais precisos, porém mais lentos. |
| `--optimize-hyperparameters` | Ativa o modo de busca automática de hiperparâmetros via Optuna. |
| `--max-total-trials` | Número máximo de tentativas (trials) que o Optuna vai explorar. Mais trials = melhor resultado, porém mais tempo. |
| `--n-jobs` | Número de trials executados em paralelo. Depende dos núcleos disponíveis na sua máquina. |
| `--optimization-log-path` | Diretório onde serão salvos os logs e checkpoints dos modelos avaliados durante a otimização. Se omitido, é criado automaticamente em `<log-folder>/<algo>/<env>_<id>/optimization/`. |
| `--storage` | URI do banco de dados onde o estudo Optuna será persistido. Use `sqlite:///nome_do_arquivo.db` para SQLite local. |
| `--study-name` | Nome único do estudo no banco de dados. Permite múltiplos estudos no mesmo arquivo `.db` e retomada após interrupção. |
 
> 💡 **Recomendação:** Não informe `--optimization-log-path` e deixe o valor padrão ser usado. Assim, os trials ficam sempre dentro de `<exp-dir>/optimization/`, que é exatamente a estrutura esperada pelo script `extract_best_trial.py` sem nenhuma configuração adicional.

#### 🔍 Inspecionando o banco de dados do Optuna

Você pode visualizar e consultar os estudos salvos usando o **dashboard do Optuna** para visualização interativa:

```bash
optuna-dashboard sqlite:///optuna_studies.db
```

Acesse no navegador: `http://localhost:8080`

---

### 3️⃣ Treinamento do Modelo

#### 1º Passo: Definição dos hiperparâmetros do PPO que seão utilizados no treinamento 

**Opção 1:** Caso você não tenha realizado o tuning de hiperparâmetros, use os valores padrão do PPO, eles são definidos por default (os parâmetros padrão do PPO estão disponíveis na [documentação oficial](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html)). Mas tenha certeza que o seu ambiente está registrado no `rl-baselines3-zoo/hyperparams/ppo.yml` para que o script de treinamento os reconheça. Caso você tenha adicionado um novo cenário ou objetivo de recompensa ao `food_delivery_gym` desde a última vez que o `ppo.yml` foi atualizado, execute o script `update_ppo_envs.py` na raiz do `rl-baselines3-zoo` antes de prosseguir. 

```bash
python scripts/update_ppo_envs.py
```

**Opção 2:** Se você realizou o tuning de hiperparâmetros usando o Optuna. Utilize o script `extract_best_trial` para identificar o melhor trial e gerar automaticamente o bloco de hiperparâmetros num arquivo `ppo.yml` separado. Esse script gera um bloco de hiperparâmetros no formato YAML como esse:
 
```yml
FoodDelivery-medium-obj1-v1:
  n_timesteps: 18000000
  policy: 'MultiInputPolicy'
  n_envs: 4
  learning_rate: 0.0009742009357947689
  ent_coef: 9.53697192932737e-07
  clip_range: 0.3
  n_steps: 256
  batch_size: 16
  n_epochs: 1
  gamma: 0.9912696235841435
  gae_lambda: 0.9936681407681269
  max_grad_norm: 1.92598340602166
  policy_kwargs: "dict(net_arch=dict(pi=[64], vf=[64]), activation_fn=nn.Tanh)"
  normalize: true
```
 
Passe o diretório do experimento gerado pelo RL Baselines3 Zoo (formato `<log-folder>/<algo>/<env>_<id>`):
 
```bash
python scripts/extract_best_trial.py \
  --exp-dir logs/ppo/FoodDelivery-medium-obj1-v1_1

#   O ambiente é inferido automaticamente a partir do diretório, mas pode ser 
# informado explicitamente se necessário
python scripts/extract_best_trial.py \
  --exp-dir logs/ppo/FoodDelivery-medium-obj1-v1_1 \
  --env FoodDelivery-medium-obj1-v1
```
 
Opções de configuração do script `extract_best_trial.py`:
 
| Opção | Descrição | Padrão |
|-------|-----------|--------|
| `--exp-dir` / `-e` | Diretório do experimento gerado pelo RL Baselines3 Zoo (ex: `logs/ppo/FoodDelivery-medium-obj1-v1_1`). A pasta `optimization/` é localizada automaticamente dentro dele. Sempre obrigatório. | — |
| `--env` | ID completo do ambiente Gymnasium (ex: `FoodDelivery-medium-obj1-v1`). Se omitido, é inferido a partir do nome do `--exp-dir`. | inferido |
| `--n-timesteps` | Timesteps para o treinamento final registrado no YAML. | `18000000` |
| `--n-envs` | Número de ambientes paralelos registrado no YAML. | `4` |
| `--output-dir` | Diretório base de saída. O subdiretório de versão do pacote é criado automaticamente. | `hyperparams/best_params_for_food_delivery_gym` |
 
O script identifica o melhor trial varrendo todos os `trial_N/evaluations.npz` dentro de `optimization/` e selecionando o trial com maior recompensa média no último checkpoint de avaliação (o mesmo critério ao usado pelo Optuna internamente). A normalização é detectada automaticamente pela presença do arquivo `report_*.pkl` na raiz do `--exp-dir`, gerado pelo Zoo ao final da otimização.
 
O script exibe um resumo completo no terminal e **adiciona** o bloco de hiperparâmetros ao arquivo `ppo.yml` sem sobrescrever entradas anteriores. O subdiretório de saída é determinado automaticamente pela versão instalada do `food_delivery_gym` (ex: `v1.2.x`).

#### 2º Passo: Execute o treinamento

Utilize os comando abaixos para iniciar o treinamento do agente PPO utilizando os hiperparâmetros extraídos no passo anterior.

```bash
# Caso queira treinar com os hiperparâmetros padrão do PPO (sem arquivo YAML)
python train.py \
  --algo ppo \
  --env FoodDelivery-medium-obj1-v1 \
  --n-timesteps 18000000 \
  --log-folder logs/training/

# Treinar com os melhores hiperparâmetros extraídos (certifique-se de que o caminho do conf-file está correto)
python train.py \
  --algo ppo \
  --env FoodDelivery-medium-obj1-v1 \
  --conf-file hyperparams/best_params_for_food_delivery_gym/v1.2.x/ppo.yml \
  --n-timesteps 18000000 \
  --log-folder logs/training/
```

> Dica: Caso você não tenha conseguido um resultado bom tente aumentar o número de timesteps para o treinamento ou tente realizar o passo de ajuste os hiperparâmetros e veja o impacto no desempenho do agente.

<!-- 
TODO -> Verificar se isso funciona
#### 🔁 Retomando um treinamento interrompido

Se o treinamento for interrompido (queda de energia, erro, etc.), especifique o último checkpoint salvo com `--trained-agent`:

```bash
# Retomar a partir do checkpoint mais recente salvo em logs/
python train.py \
  --algo ppo \
  --env FoodDelivery-medium-obj1-v1 \
  --conf-file hyperparams/best_params_for_food_delivery_gym/v1.2.x/ppo.yml \
  --n-timesteps 18000000 \
  --log-folder logs/training/ \
  --trained-agent logs/training/ppo/FoodDelivery-medium-obj1-v1_1/best_model.zip
``` -->

##### 📋 Parâmetros de treinamento explicados

| Parâmetro | Descrição |
|-----------|-----------|
| `--algo ppo` | Algoritmo utilizado para o treinamento. |
| `--env` | ID do ambiente registrado. |
| `--conf-file` | Caminho para o arquivo YAML com os hiperparâmetros. |
| `--n-timesteps` | Total de passos de treinamento. Quanto mais passos, maior o tempo e potencialmente melhor o agente. |
| `--log-folder` | Diretório onde os logs, checkpoints e o modelo final serão salvos. |
| `--trained-agent` | Caminho para um modelo `.zip` já treinado, para **continuar o treinamento** a partir de um checkpoint. |
| `--eval-episodes` | Número de episódios usados para avaliar o agente durante o treinamento (padrão: 5). |
| `--eval-freq` | Frequência (em passos) com que o agente é avaliado e um checkpoint é salvo. |
| `--save-freq` | Frequência (em passos) para salvar checkpoints intermediários. |
| `--verbose` | Nível de verbosidade: `0` = silencioso, `1` = INFO. |
| `--seed` | Seed para reprodutibilidade dos experimentos. |

##### 🔹 Exemplos adicionais de treinamento

```bash
# Treinamento com avaliação frequente e seed fixo (reprodutibilidade)
python train.py \
  --algo ppo \
  --env FoodDelivery-medium-obj1-v1 \
  --conf-file hyperparams/best_params_for_food_delivery_gym/v1.2.x/ppo.yml \
  --n-timesteps 18000000 \
  --eval-freq 10000 \
  --eval-episodes 10 \
  --seed 42 \
  --log-folder logs/training/

# Treinamento com salvamento frequente de checkpoints
python train.py \
  --algo ppo \
  --env FoodDelivery-medium-obj1-v1 \
  --conf-file hyperparams/best_params_for_food_delivery_gym/v1.2.x/ppo.yml \
  --n-timesteps 18000000 \
  --save-freq 500000 \
  --log-folder logs/training/
```

#### 3º Passo: Visualize a curva de aprendizado após o treinamento:

```bash
python scripts/plot_train.py -a ppo -e FoodDelivery-medium-obj1-v1 -f logs/training/
```

<!-- 
TODO -> Verificar a utilidade desse item
### 5️⃣ Dicas Gerais de Uso do RL Baselines3 Zoo

- **Organize os estudos por nome**: Use nomes descritivos como `ppo_medium_obj1_run1` para facilitar a rastreabilidade entre múltiplos experimentos.
- **Use o mesmo banco SQLite para tudo**: Centralizar todos os estudos (tuning e treinamento) em um único `optuna_studies.db` facilita comparações e consultas.
- **Sempre salve com `--storage`**: Mesmo para experimentos curtos, a persistência no banco de dados protege contra perdas inesperadas.
- **Verifique os logs regularmente**: O TensorBoard pode ser usado para acompanhar métricas em tempo real:

```bash
tensorboard --logdir logs/training/
```

Acesse em: `http://localhost:6006` -->

---

## 🧩 Criação de Agentes Otimizadores com `OptimizerGym`

O pacote `food_delivery_gym` permite a criação de agentes otimizadores personalizados, baseados em heurísticas ou modelos treinados com Aprendizado por Reforço (AR), por meio da classe abstrata `OptimizerGym`.

Essa abordagem é útil para avaliar o desempenho de algoritmos customizados no ambiente de entrega de última milha e comparar com agentes baseados em AR.

Para desenvolver um novo agente deve herdar `OptimizerGym` e implementar o método `select_driver`.

### 🔧 Implementando um Otimizador Heurístico

A seguir, um exemplo de implementação do otimizador baseado na **distância ao motorista mais próximo**:

```python
from food_delivery_gym.main.optimizer.optimizer_gym.optmizer_gym import OptimizerGym
from food_delivery_gym.main.route.route import Route
from food_delivery_gym.main.driver.driver import Driver
from food_delivery_gym.main.map.map import Map
from typing import List

class NearestDriverOptimizerGym(OptimizerGym):
    def get_title(self):
        return "Otimizador do Motorista Mais Próximo"

    def compare_distance(self, map: Map, driver: Driver, route: Route):
        return map.distance(driver.get_last_valid_coordinate(), route.route_segments[0].coordinate)

    def select_driver(self, obs: dict, drivers: List[Driver], route: Route):
        nearest_driver = min(drivers, key=lambda driver: self.compare_distance(self.gym_env.simpy_env.map, driver, route))
        return drivers.index(nearest_driver)
```

### 🤖 Implementando um Otimizador com Modelo de AR (PPO)

Se você já treinou um modelo com o RL Baselines3 Zoo (como mostrado na seção anterior), pode integrá-lo diretamente:

```python
class RLModelOptimizerGym(OptimizerGym):

    def __init__(self, environment: Union[FoodDeliveryGymEnv, VecEnv], model: BaseAlgorithm):
        super().__init__(environment)
        self.model = model
        
        model_env = model.get_env()
        if model_env is not None:
            print(f"Ambiente do modelo: {type(model_env).__name__}")
            print(f"Ambiente fornecido: {type(environment).__name__}")
            
            if type(model_env) != type(environment):
                print("AVISO: O ambiente fornecido é diferente do ambiente do modelo!")
                print("Isso pode causar problemas de normalização ou formato de observação.")

    def get_title(self):
        return "Otimizador por Aprendizado por Reforço"

    def select_driver(self, obs: dict, drivers: List[Driver], route: Route):
        if self.is_vectorized:
            action, _states = self.model.predict(obs, deterministic=True)
            if isinstance(action, np.ndarray):
                action = action[0] if len(action.shape) > 0 else action.item()
        else:
            action, _states = self.model.predict(obs, deterministic=True)
            if isinstance(action, np.ndarray):
                action = action.item() if action.size == 1 else action
            
        return action
```

### ▶️ Executando Simulações com o Otimizador

Com seu otimizador implementado, basta instanciá-lo e chamar:

```python
optimizer = NearestDriverOptimizerGym(env)
# ou
optimizer = RLModelOptimizerGym(env, trained_model)

optimizer.run_simulations(
      num_runs=10, dir_path="./resultados/", seed=42,
      save_individual_plots=True, save_mean_plots=True,
      metrics_fmt="npz",
  )
```

Isso executará múltiplas simulações, coletará estatísticas (recompensa, tempo de entrega, distância percorrida, etc.) e salvará os resultados em um relatório (`.txt`) e um arquivo de métricas e um arquivo `.npz` ou `.json`.

## 🧪 Scripts Utilitários

O projeto fornece quatro scripts utilitários para execução de simulações em lote, geração de gráficos, conversão de métricas e consolidação de resultados em planilhas Excel.

---

### 🚀 Script `run_batch_eval`: Execução de Múltiplos Otimizadores

Automatiza a execução de diferentes agentes otimizadores em combinações de cenários experimentais e objetivos de recompensa.

#### ✅ O que ele faz:

* Executa os seguintes agentes heurísticos:
  * `RandomDriverOptimizerGym`
  * `FirstDriverOptimizerGym`
  * `NearestDriverOptimizerGym`
  * `LowestCostDriverOptimizerGym` (com custo de rota)
  * `LowestCostDriverOptimizerGym` (com custo marginal de rota)
* Executa modelos de algoritmos RL (`RLModelOptimizerGym`), com **descoberta automática** dos modelos disponíveis em `--model-base-dir`
* Suporta dois modos de experimento para seleção dos modelos RL: `cross_scenario` e `same_scenario`
* Gera arquivos `.txt` com os resultados das execuções
* Gera arquivos `.npz` ou `.json` contendo as métricas agregadas para análise

#### 📦 Como usar:

```bash
# Execução padrão (todos os objetivos, cenários, heurísticas e modelos disponíveis)
python -m scripts.run_batch_eval
```

#### ⚙️ Opções de Configuração

| Opção | Descrição | Padrão |
|-------|-----------|--------|
| `--objectives` / `-o` | Objetivos de recompensa a executar (1–13). Aceita múltiplos valores. | todos (1–13) |
| `--scenarios` / `-s` | Cenários a executar: `simple`, `medium`, `complex`. Aceita múltiplos valores. | todos |
| `--heuristics` | Heurísticas a executar. Aceita múltiplos valores. | todas |
| `--models` / `-m` | Nomes dos modelos RL (subdiretórios de `obj_N/` com `best_model.zip`). | descoberta automática |
| `--no-heuristics` | Desativa a execução de todas as heurísticas. | — |
| `--no-rl` | Desativa a execução dos modelos PPO. | — |
| `--num-runs` / `-n` | Número de simulações por agente. | `20` |
| `--seed` | Seed para reprodutibilidade. | `123456789` |
| `--experiment-mode` | Modo de seleção dos modelos RL: `cross_scenario` ou `same_scenario`. Ver seção abaixo. | `cross_scenario` |
| `--train-scenario` | Cenário cujos modelos serão usados no modo `cross_scenario`. Ignorado em `same_scenario`. | `medium` |
| `--model-base-dir` | Diretório raiz dos modelos PPO treinados. Os modelos são buscados em `<model-base-dir>/<scenario>/treinamento/`. | `./data/ppo_training/` |
| `--results-base-dir` | Diretório base para salvar resultados. Use `{}` como placeholder para objetivo e cenário. | `./data/runs/execucoes/obj_{}/{}_scenario/` |
| `--batch-plots` | Ativa a geração de gráficos agregados (lote) ao final de cada agente. | — |
| `--all-plots` | Ativa todos os gráficos: equivale a `--batch-plots` mais gráficos individuais por episódio. | — |
| `--metrics-fmt` | Formato do arquivo de métricas: `npz` (comprimido) ou `json` (legível). | `npz` |

Os valores possíveis para `--heuristics` são: `random`, `first_driver`, `nearest_driver`, `lowest_route_cost`, `lowest_marginal_route_cost`.

> 💡 **Dica:** Para gerar gráficos de execuções anteriores sem re-executar as simulações, use o script `generate_plots` descrito abaixo.

---

#### 🔬 Modos de Experimento para Modelos RL

O argumento `--experiment-mode` controla qual modelo treinado é carregado para cada cenário de avaliação. Existem dois modos:

| Modo | Descrição |
|------|-----------|
| `cross_scenario` | Usa os modelos treinados em um único cenário fixo (definido por `--train-scenario`) para avaliar em **todos** os cenários selecionados. Útil para medir a capacidade de generalização do agente. |
| `same_scenario` | Usa o modelo treinado no **próprio cenário** de avaliação. O argumento `--train-scenario` é ignorado neste modo. |

Em ambos os modos, o caminho efetivo dos modelos é resolvido como:
```
<model-base-dir>/<train_scenario>/<model-subdir>/obj_N/<model_name>/best_model.zip
```

onde `train_scenario` é `--train-scenario` no modo `cross_scenario`, ou o próprio cenário de avaliação no modo `same_scenario`.

---

#### 📁 Estrutura de diretórios para modelos RL

O script descobre automaticamente os modelos disponíveis varrendo o diretório resolvido conforme o modo de experimento. Para que um modelo seja reconhecido, os arquivos devem estar organizados da seguinte forma:
```
<model-base-dir>/
├── medium/
│   └── treinamento/
│       ├── obj_1/
│       │   ├── 18M_steps/
│       │   │   ├── best_model.zip
│       │   │   └── FoodDelivery-medium-obj1-v1/
│       │   │       └── vecnormalize.pkl
│       │   └── outro_experimento/
│       │       ├── best_model.zip
│       │       └── FoodDelivery-medium-obj1-v1/
│       │           └── vecnormalize.pkl
│       └── obj_2/
│           └── ...
├── simple/
│   └── treinamento/
│       └── obj_1/ ...
└── complex/
    └── treinamento/
        └── obj_1/ ...
```

As regras são:

* Cada cenário deve ter um subdiretório `treinamento` em `<model-base-dir>/`
* Dentro de `<scenario>/treinamento/obj_N/`, qualquer subdiretório que contenha best_model.zip na raiz é detectado como um modelo, o nome do diretório que contém o modelo se torna o identificador nos resultados
* O `vecnormalize.pkl` é carregado automaticamente se encontrado em qualquer subdiretório do modelo. Caso contrário, o modelo é executado sem normalização

> **Observação:** se os modelos foram treinados com o RL Baselines3 Zoo sem mover os arquivos gerados, a estrutura já estará no formato correto automaticamente. Use `--models` apenas para restringir a execução a modelos específicos dentro de `obj_N/`.

#### 🔹 Exemplos

```bash
# Rodar apenas heurísticas, sem PPO
python -m scripts.run_batch_eval --no-rl

# Rodar apenas os modelos PPO, sem heurísticas
python -m scripts.run_batch_eval --no-heuristics

# Cenário e objetivo específicos
python -m scripts.run_batch_eval --scenarios medium --objectives 1 3 5

# Selecionar heurísticas específicas
python -m scripts.run_batch_eval --heuristics random nearest_driver

# Forçar modelos RL específicos (sem descoberta automática)
python -m scripts.run_batch_eval --models 18M_steps 100M_steps

# Experimento cross_scenario: modelo treinado em 'medium', avaliado em todos os cenários
python -m scripts.run_batch_eval --experiment-mode cross_scenario --train-scenario medium

# Experimento same_scenario: cada cenário usa seu próprio modelo treinado
python -m scripts.run_batch_eval --experiment-mode same_scenario

# Execução rápida para testes
python -m scripts.run_batch_eval --num-runs 5 --seed 42 --scenarios simple --objectives 1

# Gerar gráficos de lote ao final de cada agente
python -m scripts.run_batch_eval --batch-plots

# Gerar todos os gráficos (individuais + lote)
python -m scripts.run_batch_eval --all-plots

# Salvar métricas em JSON (legível) em vez de NPZ
python -m scripts.run_batch_eval --metrics-fmt json

# Execução completa com diretórios customizados
python -m scripts.run_batch_eval \
    --model-base-dir ./meus_modelos \
    --results-base-dir ./resultados/obj_{}/{}_scenario/
```

---

### 📈 Script `generate_plots`: Geração de Gráficos a partir de Métricas

Gera gráficos de episódios individuais e/ou agregados (lote) a partir dos arquivos de métricas produzidos pelo `run_batch_eval`, sem precisar re-executar as simulações.

#### ✅ O que ele faz:

* Varre os diretórios de cada agente procurando por arquivos de métricas, priorizando `metrics_data.npz` e usando `metrics_data.json` como fallback
* Gera os gráficos dentro do diretório `figs/` de cada agente, replicando exatamente a estrutura do `run_batch_eval`:

```
<results_dir>/obj_N/<scenario>_scenario/<agent>/
    figs/
        run_1_results_<reward>/               ← episódios individuais
            order_generation.png
            route_reordering.png
            driver_establishment_metrics.png
        run_2_results_<reward>/
            ...
        mean_results_<avg_reward>_route_reordering.png   ← lote
        mean_results_<avg_reward>_other_metrics.png      ← lote
```

#### 📦 Como usar:

```bash
# Gerar todos os gráficos (padrão)
python -m scripts.generate_plots --results-dir ./data/runs/execucoes

# Somente gráficos de episódios individuais
python -m scripts.generate_plots --results-dir ./data/runs/execucoes --only-episode

# Somente gráficos agregados de lote
python -m scripts.generate_plots --results-dir ./data/runs/execucoes --only-batch
```

#### ⚙️ Opções de Configuração

| Opção | Descrição | Padrão |
|-------|-----------|--------|
| `--results-dir` / `-r` | Diretório raiz com os resultados (`obj_N/`). | `./data/runs/execucoes` |
| `--objectives` / `-o` | Objetivos a processar (1–13). Aceita múltiplos valores. | todos (1–13) |
| `--scenarios` / `-s` | Cenários a processar. Aceita múltiplos valores. | todos |
| `--only-episode` | Gera somente os gráficos de episódios individuais. Mutuamente exclusivo com `--only-batch`. | — |
| `--only-batch` | Gera somente os gráficos agregados de lote. Mutuamente exclusivo com `--only-episode`. | — |

---

### 🔄 Script `convert_metrics`: Conversão entre NPZ e JSON

Converte arquivos de métricas entre os formatos NPZ (comprimido) e JSON (legível), tanto para arquivos individuais quanto para diretórios inteiros de forma recursiva.

#### ✅ O que ele faz:

* Converte um único arquivo `.npz` → `.json` ou `.json` → `.npz`
* Quando recebe um diretório, varre recursivamente procurando por todos os arquivos com o nome-base configurado (padrão: `metrics_data`) e converte cada um no mesmo diretório onde foi encontrado
* Suporta `--dry-run` para inspecionar o que seria convertido sem gravar nada

#### 📦 Como usar:

```bash
# Converter um arquivo NPZ individual para JSON
python -m scripts.convert_metrics ./data/runs/execucoes/obj_1/simple_scenario/random/metrics_data.npz

# Converter um arquivo JSON individual para NPZ
python -m scripts.convert_metrics ./data/runs/execucoes/obj_1/simple_scenario/random/metrics_data.json

# Converter todos os arquivos metrics_data.npz e metrics_data.json em um diretório (recursivo)
python -m scripts.convert_metrics ./data/runs/execucoes

# Verificar o que seria convertido sem gravar nada
python -m scripts.convert_metrics ./data/runs/execucoes --dry-run
```

#### ⚙️ Opções de Configuração

| Opção | Descrição | Padrão |
|-------|-----------|--------|
| `path` | Arquivo `.npz` / `.json` ou diretório a processar. | — |
| `--name` | Nome-base dos arquivos buscados em modo diretório. | `metrics_data` |
| `--dry-run` | Exibe o que seria convertido sem gravar nenhum arquivo. | — |

---

### 📊 Script `generate_table`: Geração de Planilhas Excel com Métricas

Consolida os resultados gerados pelo `run_batch_eval` e gera automaticamente uma planilha Excel com as métricas estatísticas de todos os agentes. Suporta métricas em NPZ ou JSON, priorizando NPZ quando ambos estiverem disponíveis.

#### ✅ O que ele faz:

* Varre o diretório de resultados e **descobre automaticamente** todos os agentes presentes — sem mapeamentos manuais de colunas
* Organiza os agentes em ordem: heurísticas conhecidas primeiro, modelos PPO em seguida (ordem alfabética)
* Gera o Excel do zero com estrutura dinâmica: novas heurísticas ou modelos PPO geram colunas novas automaticamente
* Preenche três abas com média, desvio padrão, mediana e moda:
  * **Recompensas**
  * **Tempo Efetivo Gasto**
  * **Distância Percorrida**
* Destaca em negrito o melhor agente por cenário/objetivo em cada aba (maior média em Recompensas; menor nas demais)
* Cria automaticamente os diretórios de saída caso não existam

#### 📦 Como usar:

```bash
# Padrão — gera tabela com todos os dados disponíveis
python -m scripts.generate_table
```

#### ⚙️ Opções de Configuração

| Opção | Descrição | Padrão |
|-------|-----------|--------|
| `--results-dir` / `-r` | Diretório raiz com os resultados (`obj_N/`). | `./data/runs/execucoes` |
| `--output` / `-out` | Caminho do arquivo Excel de saída. Diretórios intermediários são criados automaticamente. | `./data/runs/tabelas/objective_table.xlsx` |
| `--objectives` / `-o` | Objetivos a incluir (1–13). Aceita múltiplos valores. | todos (1–13) |
| `--scenarios` / `-s` | Cenários a incluir: `simple`, `medium`, `complex`. Aceita múltiplos valores. | todos |

#### 🔹 Exemplos

```bash
# Diretório e arquivo de saída customizados
python -m scripts.generate_table \
    --results-dir ./data/runs/execucoes \
    --output ./data/runs/tabelas/objective_table.xlsx

# Apenas objetivos e cenários específicos
python -m scripts.generate_table --objectives 1 3 5 --scenarios simple medium
```

---

### 📊 Script `generate_boxplots`: Geração de Boxplots Comparativos

Gera boxplots para comparar visualmente o desempenho dos agentes em três métricas independentes: **Recompensa Acumulada**, **Tempo Efetivo Gasto** e **Distância Percorrida**. Lê os arquivos `metrics_data.npz` / `metrics_data.json` produzidos pelo `run_batch_eval`.

#### O que ele faz:

* Descobre automaticamente todos os agentes com dados disponíveis no diretório de resultados
* Suporta dois modos de agrupamento no eixo X: **por agente** (padrão) ou **por cenário** (`--by-scenario`)
* Permite gerar uma figura única com as três métricas lado a lado, arquivos separados por métrica (`--split`) ou um subplot por cenário (`--split-scenarios`)
* Exibe opcionalmente outliers, médias (losango), valores numéricos das médias e anotação do N amostral por box
* Adiciona entradas de estatísticas (mediana e média) na legenda via `--legend-stats`
* Exporta nos formatos `pdf`, `png` ou `svg`

#### 📦 Como usar:

```bash
# Gerar figura com os três boxplots (padrão: objetivo 3, todos os cenários)
python generate_boxplots.py

# Selecionar agentes e cenário específico
python generate_boxplots.py --agents random nearest_driver lowest_marginal_route_cost \
       --scenarios simple

# Comparar dois modelos PPO com heurísticas, objetivo 5
python generate_boxplots.py --agents random nearest_driver ppo_18M ppo_36M --objective 5

# Salvar cada métrica em arquivo separado, alta resolução, formato SVG
python generate_boxplots.py --split --dpi 600 --fmt svg

# Um PNG por métrica com N subplots (um por cenário), legenda unificada
python generate_boxplots.py --by-scenario --split-scenarios

# Ocultar outliers, exibir médias e anotar N amostras
python generate_boxplots.py --no-fliers --show-means --show-mean-values --annotate-n

# Adicionar entradas de mediana e média na legenda
python generate_boxplots.py --by-scenario --split-scenarios --show-means --legend-stats

# Selecionar apenas recompensa e distância
python generate_boxplots.py --metrics rewards distance
```

#### ⚙️ Opções de Configuração

**Dados**

| Opção | Descrição | Padrão |
|-------|-----------|--------|
| `--results-dir` / `-r` | Diretório raiz com os resultados (`obj_N/`). | `./data/runs/execucoes` |
| `--objective` / `-o` | Número do objetivo a processar (`obj_N/`). | `3` |
| `--scenarios` / `-s` | Cenários a incluir: `simple`, `medium`, `complex`. Aceita múltiplos valores. | todos |
| `--agents` / `-a` | Nomes dos diretórios dos agentes, na ordem desejada. Se omitido, todos os agentes com dados são descobertos automaticamente. | descoberta automática |
| `--exclude-agents` | Agentes a excluir mesmo que descobertos automaticamente. | — |

**Métricas**

| Opção | Descrição | Padrão |
|-------|-----------|--------|
| `--metrics` / `-M` | Métricas a plotar: `rewards`, `delivery_time`, `distance`. Aceita múltiplos valores. | todas |

**Layout e visualização**

| Opção | Descrição | Padrão |
|-------|-----------|--------|
| `--by-scenario` | Agrupa por agente no eixo X, usando cenários como grupos de boxes. Por padrão, o eixo X é de cenários com agentes como grupos. | — |
| `--split-scenarios` | Gera 1 arquivo por métrica com N subplots (um por cenário), cada um com escala Y independente. Requer `--by-scenario`. | — |
| `--split` | Salva cada métrica em um arquivo separado em vez de uma figura única. | — |
| `--no-fliers` | Oculta os outliers (pontos além dos whiskers). | — |
| `--show-means` | Exibe a média como um losango dentro de cada box. | — |
| `--show-mean-values` | Anota o valor numérico da média ao lado de cada losango. Requer `--show-means`. | — |
| `--annotate-n` | Anota o número de amostras (N) abaixo de cada box. | — |
| `--suptitle` | Título geral da figura. | — |
| `--legend-cols` | Número de colunas da legenda. | automático |
| `--legend-stats` | Adiciona entradas de Mediana (linha preta) e, se `--show-means` ativo, Média (losango) na legenda. | — |

**Saída**

| Opção | Descrição | Padrão |
|-------|-----------|--------|
| `--output-dir` / `-od` | Diretório de saída das figuras. | `./data/runs/figuras` |
| `--prefix` | Prefixo do nome do arquivo de saída. | `boxplot` |
| `--fmt` | Formato de saída: `pdf`, `png`, `svg`. | `pdf` |
| `--figsize` | Dimensões da figura `(largura altura)` em polegadas. | `16 5` |
| `--dpi` | Resolução em DPI. | `300` |
| `--font-size` | Tamanho base da fonte. | `9` |