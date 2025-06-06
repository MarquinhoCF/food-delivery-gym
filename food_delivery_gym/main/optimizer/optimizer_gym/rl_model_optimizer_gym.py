from typing import List, Union
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv

from food_delivery_gym.main.driver.driver import Driver
from food_delivery_gym.main.environment.food_delivery_gym_env import FoodDeliveryGymEnv
from food_delivery_gym.main.optimizer.optimizer_gym.optmizer_gym import OptimizerGym
from food_delivery_gym.main.route.route import Route


class RLModelOptimizerGym(OptimizerGym):

    def __init__(self, environment: Union[FoodDeliveryGymEnv, VecEnv], model: PPO):
        super().__init__(environment)
        self.model = model
        
        # Verifica se o ambiente do modelo é compatível
        model_env = model.get_env()
        if model_env is not None:
            print(f"Ambiente do modelo: {type(model_env).__name__}")
            print(f"Ambiente fornecido: {type(environment).__name__}")
            
            # Se o modelo tem um ambiente diferente, avisa
            if type(model_env) != type(environment):
                print("AVISO: O ambiente fornecido é diferente do ambiente do modelo!")
                print("Isso pode causar problemas de normalização ou formato de observação.")

    def get_title(self):
        return "Otimizador por Aprendizado por Reforço"

    def select_driver(self, obs: dict, drivers: List[Driver], route: Route):
        """
        Seleciona o driver usando o modelo PPO treinado.
        
        Args:
            obs: Observação do ambiente
            drivers: Lista de drivers disponíveis
            route: Rota a ser executada
            
        Returns:
            int: Índice do driver selecionado
        """
        # Se o ambiente é vectorizado, a observação já está no formato correto
        if self.is_vectorized:
            # Para ambientes vectorizados, obs já deve estar no formato correto
            action, _states = self.model.predict(obs, deterministic=True)
            # Se action for um array, pega o primeiro elemento
            if isinstance(action, np.ndarray):
                action = action[0] if len(action.shape) > 0 else action.item()
        else:
            # Para ambientes não vectorizados, pode precisar converter
            action, _states = self.model.predict(obs, deterministic=True)
            if isinstance(action, np.ndarray):
                action = action.item() if action.size == 1 else action
            
        return action