from food_delivery_gym.main.actors.actor import Actor
from food_delivery_gym.main.base.types import Coordinate
from food_delivery_gym.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv


class MapActor(Actor):

    def __init__(self, environment: FoodDeliverySimpyEnv, coordinate: Coordinate, available: bool) -> None:
        super().__init__(environment)
        self.coordinate = coordinate
        self.available = available
