from typing import List

from food_delivery_gym.main.driver.driver import Driver
from food_delivery_gym.main.optimizer.optimizer_gym.optmizer_gym import OptimizerGym
from food_delivery_gym.main.route.route import Route


class RandomDriverOptimizerGym(OptimizerGym):

    def get_title(self):
        return "Otimizador do Motorista Aleatório"

    def ranked_actions(
        self,
        obs: dict,
        drivers: List[Driver],
        route: Route,
        *,
        rng=None,
    ) -> list[int]:
        del obs, route
        if rng is None:
            raise ValueError(
                "RandomDriverOptimizerGym.ranked_actions requer rng "
                "(não usa o RNG do episódio)"
            )
        return [int(i) for i in rng.permutation(len(drivers))]

    def select_driver(self, obs: dict, drivers: List[Driver], route: Route):
        return self.gym_env.action_space.sample()
