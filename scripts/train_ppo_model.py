import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from importlib.resources import files
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from food_delivery_gym.main.environment.env_mode import EnvMode
from food_delivery_gym.main.environment.step_reward_logger import StepRewardLogger
from food_delivery_gym.main.environment.food_delivery_gym_env import FoodDeliveryGymEnv

SEED = 101010
SAVE_LOG_TO_FILE = False
DIR_PATH = "./data/ppo_training/obj_1/medium_scenario/otimization/1M5k-timesteps_50-max-trials/24000000_time_steps_best_params/"
SCENARIO = "medium_obj1.json"

def setup_logging():
    os.makedirs(DIR_PATH, exist_ok=True)
    for subdir in ["logs", "logs_eval", "best_model", "ppo_tensorboard"]:
        os.makedirs(os.path.join(DIR_PATH, subdir), exist_ok=True)

    if SAVE_LOG_TO_FILE:
        log_file = open(os.path.join(DIR_PATH, "log.txt"), "w", encoding="utf-8")
        sys.stdout = log_file
        sys.stderr = log_file
        return log_file
    return None

def create_env(path, is_eval=False):
    env = FoodDeliveryGymEnv(scenario_json_file_path=path)
    env.set_mode(EnvMode.TRAINING)
    env.set_reward_objective(1)
    env = Monitor(env, os.path.join(DIR_PATH, "logs_eval" if is_eval else "logs"))
    env = DummyVecEnv([lambda: env])
    return VecNormalize(env, norm_obs=True, norm_reward=False)

def plot_learning_curves(log_data_path, step_reward_logger_path):
    monitor_data = pd.read_csv(log_data_path, skiprows=1)
    rewards = monitor_data["r"].values

    # Plot recompensa por episódio
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title("Recompensa por Episódio")
    plt.xlabel("Episódios")
    plt.ylabel("Recompensa")
    plt.savefig(os.path.join(DIR_PATH, "curva_de_aprendizado.png"), dpi=300, bbox_inches='tight')
    plt.show()

    # Média e desvio a cada 1000 episódios
    chunk = 1000
    means, stds = zip(*[(np.mean(rewards[i - chunk:i]), np.std(rewards[i - chunk:i])) 
                        for i in range(chunk, len(rewards), chunk)])

    plt.figure(figsize=(10, 5))
    plt.plot(means, label="Média")
    plt.fill_between(range(len(means)), np.array(means) - stds, np.array(means) + stds, alpha=0.2, label="Desvio Padrão")
    plt.title("Média e Desvio por Bloco de 1000 Episódios")
    plt.xlabel("Episódios (x1000)")
    plt.ylabel("Retornos")
    plt.legend()
    plt.savefig(os.path.join(DIR_PATH, "curva_de_aprendizado_avg_std_1000_ep.png"), dpi=300, bbox_inches='tight')
    plt.show()

    # Recompensa por passo
    step_df = pd.read_csv(step_reward_logger_path)
    plt.figure(figsize=(12, 5))
    plt.plot(step_df["reward"], alpha=0.6, linewidth=0.7)
    plt.title("Recompensa a cada passo")
    plt.xlabel("Passos")
    plt.ylabel("Recompensa")
    plt.savefig(os.path.join(DIR_PATH, "recompensa_por_passo.png"), dpi=300, bbox_inches='tight')
    plt.show()

    # Tendência por passo dentro do episódio
    step_rewards = step_df["reward"].values
    episode_len = monitor_data["l"].values[0]  # ou gym_env.env.num_orders se necessário
    num_eps = len(step_rewards) // episode_len
    steps_matrix = step_rewards[:num_eps * episode_len].reshape((num_eps, episode_len))
    mean_per_step = steps_matrix.mean(axis=0)
    std_per_step = steps_matrix.std(axis=0)

    plt.figure(figsize=(12, 5))
    plt.plot(mean_per_step, label="Média por passo")
    plt.fill_between(range(episode_len), mean_per_step - std_per_step, mean_per_step + std_per_step, alpha=0.2)
    plt.title("Tendência da Recompensa por Passo")
    plt.xlabel("Passo")
    plt.ylabel("Recompensa")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(DIR_PATH, "tendencia_por_passo_no_episodio.png"), dpi=300, bbox_inches='tight')
    plt.show()

def main():
    log_file = setup_logging()

    scenario_path = str(files("food_delivery_gym.main.scenarios").joinpath(SCENARIO))

    # Setup dos ambientes
    train_env = create_env(scenario_path)
    eval_env = create_env(scenario_path, is_eval=True)

    check_env(train_env)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(DIR_PATH, "best_model"),
        log_path=os.path.join(DIR_PATH, "logs"),
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    step_logger = StepRewardLogger(os.path.join(DIR_PATH, "logs/step_rewards.csv"))

    best_params = {
        'batch_size': 32,
        'n_steps': 512,
        'gamma': 0.9957742128778314,
        'gae_lambda': 1 - 0.07691363050279466,
        'learning_rate': 0.0012385994626143365,
        'ent_coef': 4.346389035215521e-07,
        'clip_range': 0.2,
        'n_epochs': 1,
        'max_grad_norm': 0.5730007621571269,
        'policy_kwargs': dict(activation_fn=torch.nn.Tanh, net_arch=[dict(pi=[64], vf=[64])]),
    }

    model = PPO('MultiInputPolicy', train_env, verbose=1, **best_params)

    print("Iniciando treinamento...")
    start = time.time()
    model.learn(total_timesteps=24_000_000, callback=[eval_callback, step_logger])
    duration = time.time() - start
    formatted = time.strftime("%H:%M:%S", time.gmtime(duration))
    print(f"Tempo total de treinamento: {formatted}")

    with open(os.path.join(DIR_PATH, "training_time.txt"), "w", encoding="utf-8") as f:
        f.write(f"Tempo total de treinamento: {formatted}\n")

    model.save(os.path.join(DIR_PATH, "final_model"))

    plot_learning_curves(
        log_data_path=os.path.join(DIR_PATH, "logs/monitor.csv"),
        step_reward_logger_path=os.path.join(DIR_PATH, "logs/step_rewards.csv")
    )

    if log_file:
        log_file.close()

if __name__ == "__main__":
    main()
