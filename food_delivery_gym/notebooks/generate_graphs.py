import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
from food_delivery_gym.main.utils.load_scenarios import load_scenario
from food_delivery_gym.main.environment.food_delivery_gym_env import FoodDeliveryGymEnv

DIR_PATH = "./data/ppo_training/obj_3/complex_scenario/30000000_time_steps/"

gym_env: FoodDeliveryGymEnv = load_scenario("complex.json")

# Carregar os dados do Monitor para análise
log_data = pd.read_csv(os.path.join(DIR_PATH, "logs/monitor.csv"), skiprows=1)

# Plotar recompensa acumulada por episódio
retornos = log_data["r"].values 
plt.figure(figsize=(10, 5))
plt.plot(retornos, label="Recompensa")
plt.xlabel("Episódios")
plt.ylabel("Recompensa")
plt.title("Curva de Aprendizado - Recompensa por Episódio")
plt.legend()
plt.show()

# Calcular a média e o desvio padrão a cada mil episódios
media_1000_episodios = []
desvio_1000_episodios = []
for i in range(1000, len(retornos), 1000):
    media_1000_episodios.append(np.mean(retornos[i-1000:i]))
    desvio_1000_episodios.append(np.std(retornos[i-1000:i]))
media_1000_episodios = np.array(media_1000_episodios)
desvio_1000_episodios = np.array(desvio_1000_episodios)

# Plotar a curva de aprendizado com a média e o desvio padrão
plt.figure(figsize=(10, 5))
plt.plot(media_1000_episodios, label="Média a cada 1000 episódios")
plt.fill_between(range(len(media_1000_episodios)), media_1000_episodios - desvio_1000_episodios, media_1000_episodios + desvio_1000_episodios, alpha=0.2, label="Desvio Padrão")
plt.title('Curva de Aprendizado (média e desvio padrão a cada mil episódios)')
plt.xlabel('Episódios (x1000)')
plt.ylabel('Retornos')
plt.legend()
plt.show()

# Plotar a recompensa acumulada por passo
step_rewards_df = pd.read_csv(os.path.abspath(os.path.join(DIR_PATH, "logs/step_rewards.csv")))

# Plotar a tendência de recompensa ao longo do episódio
step_rewards = step_rewards_df["reward"].values

episode_length = gym_env.num_orders

# Quebrar o vetor em episódios (cada um com EPISODE_LENGTH passos)
num_episodes = len(step_rewards) // episode_length
step_matrix = step_rewards[:num_episodes * episode_length].reshape((num_episodes, episode_length))

# Calcular a média e desvio padrão em cada posição do episódio
mean_rewards = step_matrix.mean(axis=0)
std_rewards = step_matrix.std(axis=0)

# Plotar
plt.figure(figsize=(12, 5))
plt.plot(mean_rewards, label="Média por passo no episódio", linewidth=2)
plt.fill_between(range(episode_length), mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2, label="Desvio Padrão")
plt.xlabel("Passo dentro do episódio")
plt.ylabel("Recompensa")
plt.title("Tendência de Recompensa ao Longo do Episódio")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.3)
plt.show()

step_matrix = step_rewards[49500*episode_length:50000*episode_length].reshape((50000-49500, episode_length))

mean_rewards = step_matrix.mean(axis=0)
std_rewards = step_matrix.std(axis=0)
min_rewards = step_matrix.min(axis=0)

# Plotar
plt.figure(figsize=(12, 5))
plt.plot(mean_rewards, label="Média por passo no episódio", linewidth=2)
plt.plot(min_rewards, label="Mínimo por passo no episódio", linewidth=1)
plt.fill_between(range(episode_length), mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2, label="Desvio Padrão")
plt.xlabel("Passo dentro do episódio")
plt.ylabel("Recompensa")
plt.title("Tendência de Recompensa ao Longo do Episódio")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.3)
plt.show()