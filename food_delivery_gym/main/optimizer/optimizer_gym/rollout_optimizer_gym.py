from typing import List, Optional, Type, Tuple

from food_delivery_gym.main.driver.driver import Driver
from food_delivery_gym.main.environment.food_delivery_gym_env import FoodDeliveryGymEnv
from food_delivery_gym.main.optimizer.optimizer_gym.optmizer_gym import OptimizerGym
from food_delivery_gym.main.optimizer.optimizer_gym.nearest_driver_optimizer_gym import NearestDriverOptimizerGym
from food_delivery_gym.main.route.route import Route


def _coord_to_list(coord) -> list[float]:
    return [float(coord[0]), float(coord[1])]


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
        record_decisions: bool = True,
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
            record_decisions: se True, grava Q-values e trajetórias de
                rollout em `decision_log` a cada chamada de select_driver.
        """
        super().__init__(environment)
        self.base_optimizer_cls = base_optimizer_cls
        self.base_optimizer_kwargs = base_optimizer_kwargs or {}
        self.alpha = alpha
        self.horizon = horizon
        self.record_decisions = record_decisions
        self.decision_log: list[dict] = []

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

    def _rollout_from(self, cloned_env: FoodDeliveryGymEnv, obs, done: bool, truncated: bool) -> Tuple[float, list[dict]]:
        """
        Executa a política de base no clone e retorna
        (valor_descontado, trajetória_de_passos).
        """
        trajectory: list[dict] = []

        if done or truncated:
            return 0.0, trajectory

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

            drivers = cloned_env.get_drivers()
            driver = drivers[action] if 0 <= action < len(drivers) else None

            obs, reward, terminated, truncated_flag, info = cloned_env.step(action)

            discounted_reward = discount * reward
            if self.record_decisions:
                trajectory.append({
                    "step": steps,
                    "action": int(action),
                    "driver_id": int(driver.driver_id) if driver is not None else None,
                    "order_id": int(order.order_id) if order is not None else None,
                    "reward": float(reward),
                    "discounted_reward": float(discounted_reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated_flag),
                })

            base_optimizer.state = obs
            base_optimizer.done = terminated
            base_optimizer.truncated = truncated_flag

            total_reward += discounted_reward
            discount *= self.alpha
            steps += 1

        return total_reward, trajectory

    # Seleção da ação (motorista) via rollout
    def select_driver(self, obs: dict, drivers: List[Driver], route: Route):
        best_action = None
        best_value = float("-inf")
        candidates: list[dict] = []

        for action in range(len(drivers)):
            cloned_env = self._clone_env()

            order_before = self.gym_env.get_current_order()
            obs_after, reward, terminated, truncated, info = cloned_env.step(action)

            rollout_value, trajectory = self._rollout_from(
                cloned_env, obs_after, terminated, truncated
            )
            q_value = reward + self.alpha * rollout_value

            if self.record_decisions:
                driver = drivers[action]
                candidates.append({
                    "action": action,
                    "driver_id": int(driver.driver_id),
                    "coord": _coord_to_list(driver.coordinate),
                    "order_id": int(order_before.order_id) if order_before is not None else None,
                    "immediate_reward": float(reward),
                    "rollout_value": float(rollout_value),
                    "q_value": float(q_value),
                    "terminated_after_action": bool(terminated),
                    "truncated_after_action": bool(truncated),
                    "trajectory": trajectory,
                })

            if q_value > best_value:
                best_value = q_value
                best_action = action

        if self.record_decisions:
            order = self.gym_env.get_current_order()
            simpy_env = self.gym_env.get_simpy_env()
            chosen_driver = drivers[best_action] if best_action is not None else None
            self.decision_log.append({
                "decision_idx": len(self.decision_log),
                "sim_time": float(simpy_env.now),
                "order_id": int(order.order_id) if order is not None else None,
                "chosen_action": best_action,
                "chosen_driver_id": (
                    int(chosen_driver.driver_id) if chosen_driver is not None else None
                ),
                "best_q": float(best_value) if best_action is not None else None,
                "alpha": float(self.alpha),
                "horizon": self.horizon,
                "candidates": candidates,
            })

        return best_action
