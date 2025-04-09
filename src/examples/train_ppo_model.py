import os
import sys
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from src.main.environment.env_mode import EnvMode
from src.main.environment.step_reward_logger import StepRewardLogger
from src.main.utils.load_scenarios import load_scenario
from src.main.environment.food_delivery_gym_env import FoodDeliveryGymEnv

SEED = 101010

# Escolha se deseja salvar o log em um arquivo
SAVE_LOG_TO_FILE = False

DIR_PATH = "./data/ppo_training/obj_3/complex_scenario/13000000_time_steps/"

# Verificar e criar os diretórios necessários
os.makedirs(DIR_PATH + "logs/", exist_ok=True)
os.makedirs(DIR_PATH + "best_model/", exist_ok=True)
os.makedirs(DIR_PATH + "ppo_tensorboard/", exist_ok=True)

if SAVE_LOG_TO_FILE:
    log_file = open(DIR_PATH + "log.txt", "w", encoding="utf-8")
    sys.stdout = log_file
    sys.stderr = log_file

def main():
    try:
        # Criando o ambiente de treinamento
        gym_env: FoodDeliveryGymEnv = load_scenario("complex.json")
        gym_env.set_mode(EnvMode.TRAINING)
        gym_env.set_reward_objective(3)

        # Verificar se o ambiente está implementado corretamente
        check_env(gym_env, warn=True)

        # Monitorando o ambiente de treinamento
        gym_env = Monitor(gym_env, DIR_PATH + "logs/")
        env = DummyVecEnv([lambda: gym_env])

        # Criando o ambiente de avaliação separado (sem Monitor)
        eval_env = load_scenario("complex.json")
        eval_env.set_mode(EnvMode.TRAINING)
        eval_env.set_reward_objective(3)
        eval_env = Monitor(eval_env, DIR_PATH + "logs_eval/")
        eval_env = DummyVecEnv([lambda: eval_env])

        # Criando o EvalCallback para salvar o melhor modelo
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=DIR_PATH + "best_model/",
            log_path=DIR_PATH + "logs/",
            eval_freq=5000,  # Avaliação a cada 5k timesteps
            deterministic=True,
            render=False
        )
        
        step_reward_logger = StepRewardLogger(DIR_PATH + "logs/step_rewards.csv")

        # Treinar o modelo com EvalCallback
        model = PPO('MultiInputPolicy', env, verbose=1)
        start_time = time.time()
        model.learn(total_timesteps=13000000, callback=[eval_callback, step_reward_logger])
        end_time = time.time()
        training_time = end_time - start_time

        # Converter segundos para hh:MM:ss
        hours = int(training_time // 3600)
        minutes = int((training_time % 3600) // 60)
        seconds = int(training_time % 60)
        formatted_time = f"{hours:02}:{minutes:02}:{seconds:02}"

        print(f"Total Training Time: {formatted_time}")

        with open(DIR_PATH + "training_time.txt", "w", encoding="utf-8") as time_training_file:
            time_training_file.write(f"Tempo total de treinamento: {formatted_time}\n")

        # Salvar o modelo final
        model.save(DIR_PATH + "final_model")

        # Carregar os dados do Monitor para análise
        log_data = pd.read_csv(DIR_PATH + "logs/monitor.csv", skiprows=1)

        # Plotar recompensa acumulada por episódio
        retornos = log_data["r"].values 
        plt.figure(figsize=(10, 5))
        plt.plot(retornos, label="Recompensa")
        plt.xlabel("Episódios")
        plt.ylabel("Recompensa")
        plt.title("Curva de Aprendizado - Recompensa por Episódio")
        plt.legend()
        plt.savefig(DIR_PATH + "curva_de_aprendizado.png", dpi=300, bbox_inches='tight')
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
        plt.savefig(DIR_PATH + "curva_de_aprendizado_avg_std_1000_ep.png", dpi=300, bbox_inches='tight')
        plt.show()

        # Plotar a recompensa acumulada por passo
        step_rewards_df = pd.read_csv(DIR_PATH + "logs/step_rewards.csv")

        plt.figure(figsize=(12, 5))
        plt.plot(step_rewards_df["reward"], alpha=0.6, linewidth=0.7)
        plt.xlabel("Passos")
        plt.ylabel("Recompensa")
        plt.title("Recompensa a cada passo durante o treinamento")
        plt.savefig(DIR_PATH + "recompensa_por_passo.png", dpi=300, bbox_inches='tight')
        plt.show()

        # Plotar a tendência de recompensa ao longo do episódio
        step_rewards = step_rewards_df["reward"].values

        episode_length = gym_env.env.num_orders

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
        plt.savefig(DIR_PATH + "tendencia_por_passo_no_episodio.png", dpi=300, bbox_inches='tight')
        plt.show()


    except ValueError as e:
        print(e)


if __name__ == '__main__':
    main()

if SAVE_LOG_TO_FILE:
    log_file.close()
