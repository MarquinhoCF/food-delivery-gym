from typing import List, Optional, Type

from food_delivery_gym.main.driver.driver import Driver
from food_delivery_gym.main.environment.food_delivery_gym_env import FoodDeliveryGymEnv
from food_delivery_gym.main.optimizer.optimizer_gym.optmizer_gym import OptimizerGym
from food_delivery_gym.main.optimizer.optimizer_gym.nearest_driver_optimizer_gym import NearestDriverOptimizerGym
from food_delivery_gym.main.route.route import Route


class RolloutOptimizerGym(OptimizerGym):
    """
    Heurística de Rollout sobre o OptimizerGym.

    Para o pedido/estado atual, e para cada motorista candidato (ação):
      1. Clona o ambiente real (snapshot/restore SimPy), preservando-o intacto.
      2. Aplica a ação candidata no ambiente clonado (1 passo real).
      3. A partir daí, executa a política de base (base_optimizer) no
         ambiente clonado, acumulando recompensa descontada por α, até:
           - o episódio terminar (done/truncated), ou
           - atingir o horizonte `horizon` (se definido), somando nesse
             caso uma aproximação de custo terminal (TODO).
      4. Escolhe a ação com o maior Q-factor estimado (reward imediato +
         α * valor do rollout) e a retorna para ser executada no ambiente
         REAL (isso é feito pelo framework, via assign_driver_to_order).
    """

    def __init__(
        self,
        environment: FoodDeliveryGymEnv,
        base_optimizer_cls: Type[OptimizerGym] = NearestDriverOptimizerGym,
        base_optimizer_kwargs: Optional[dict] = None,
        alpha: float = 1.0,
        horizon: Optional[int] = None,
    ):
        """
        Args:
            environment: ambiente real (FoodDeliveryGymEnv), não vetorizado.
            base_optimizer_cls: classe do otimizador usado como política de
                base para completar as trajetórias de rollout.
            base_optimizer_kwargs: kwargs extras para instanciar a base
                (ex.: {"cost_function": ...}).
            alpha: fator de desconto aplicado a cada passo adicional do
                rollout (alpha=1.0 -> sem desconto).
            horizon: número de passos de rollout após a ação candidata. Se
                None, o rollout roda até o episódio terminar.
        """
        super().__init__(environment)
        self.base_optimizer_cls = base_optimizer_cls
        self.base_optimizer_kwargs = base_optimizer_kwargs or {}
        self.alpha = alpha
        self.horizon = horizon

    def get_title(self):
        base_name = self.base_optimizer_cls.__name__
        horizon_str = f"H={self.horizon}" if self.horizon is not None else "H=inf"
        return f"Rollout({base_name}, alpha={self.alpha}, {horizon_str})"

    # Aproximação de custo terminal (TODO)
    def terminal_cost_to_go(self, cloned_env: FoodDeliveryGymEnv) -> float:
        """
        Aproximação do valor (recompensa) a partir do estado atual do
        ambiente clonado até o fim do episódio, usada para compensar o
        truncamento do rollout em `self.horizon` passos.

        TODO: substituir por uma aproximação real (rede neural treinada,
        heurística de custo, etc). Por enquanto retorna 0.
        """
        return 0.0

    def _clone_env(self) -> FoodDeliveryGymEnv:
        return self.gym_env.clone()

    # Rollout da política de base a partir de um estado já avançado
    def _rollout_from(self, cloned_env: FoodDeliveryGymEnv, obs, done: bool, truncated: bool) -> float:
        if done or truncated:
            return 0.0

        base_optimizer = self.base_optimizer_cls(cloned_env, **self.base_optimizer_kwargs)
        base_optimizer.state = obs
        base_optimizer.done = done
        base_optimizer.truncated = truncated

        total_reward = 0.0
        discount = self.alpha
        steps = 0

        while not (base_optimizer.done or base_optimizer.truncated):
            if self.horizon is not None and steps >= self.horizon:
                total_reward += discount * self.terminal_cost_to_go(cloned_env)
                break

            order = cloned_env.get_current_order()
            action = base_optimizer.assign_driver_to_order(base_optimizer.state, order)

            obs, reward, terminated, truncated_flag, info = cloned_env.step(action)

            base_optimizer.state = obs
            base_optimizer.done = terminated
            base_optimizer.truncated = truncated_flag

            total_reward += discount * reward
            discount *= self.alpha
            steps += 1

        return total_reward

    # Seleção da ação (motorista) via rollout
    def select_driver(self, obs: dict, drivers: List[Driver], route: Route):
        best_action = None
        best_value = float("-inf")

        for action in range(len(drivers)):
            cloned_env = self._clone_env()

            obs_after, reward, terminated, truncated, info = cloned_env.step(action)

            rollout_value = self._rollout_from(cloned_env, obs_after, terminated, truncated)
            q_value = reward + self.alpha * rollout_value

            if q_value > best_value:
                best_value = q_value
                best_action = action

        return best_action