import sys
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
import food_delivery_gym.main.environment

SEED = 101010

# Escolha se deseja salvar o log em um arquivo
SAVE_LOG_TO_FILE = False

if SAVE_LOG_TO_FILE:
    log_file = open("log.txt", "w", encoding="utf-8")
    sys.stdout = log_file
    sys.stderr = log_file

def main():
    try:
        gym_env = gym.make("FoodDelivery-medium-v0")

        estado : list[int] = gym_env.reset(seed=SEED, options={"render_mode": "human"})
        print(f'estado inicial {estado}')

        i = 1
        done = False
        truncado = False
        soma_recompensa = 0
        while not done and not truncado:
            acao = gym_env.action_space.sample() # Ação aleatória
            print("------------------> Step " + str(i) +" <------------------")
            print(f'{acao=}')
            estado, recompensa, done, truncado, info = gym_env.step(acao)
            gym_env.simpy_env.print_enviroment_state()
            print(f'estado_depois={estado}')
            print(f'{recompensa=}')
            soma_recompensa += recompensa
            i += 1
        
        print("--------------> Fim do ambiente <--------------")
        gym_env.simpy_env.print_enviroment_state()
        print(f'observação final = {gym_env.get_observation()}')
        print(f'soma das recompensas = {soma_recompensa}')
        print(f'quantidade de rotas criadas = {gym_env.simpy_env.state.get_length_orders()}')
        print(f'quantidade de rotas entregues = {gym_env.simpy_env.state.orders_delivered}')

        gym_env.show_statistcs_board()

    except ValueError as e:
        print(e)


if __name__ == '__main__':
    main()

if SAVE_LOG_TO_FILE:
    log_file.close()