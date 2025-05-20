from numbers import Number
from food_delivery_gym.main.base.dimensions import Dimensions


class Capacity:
    def __init__(self, dimensions: Dimensions) -> None:
        self.dimensions = dimensions

    def fits(self, dimensions: Dimensions) -> bool:
        return self.dimensions > dimensions

    @property
    def value(self) -> Number:
        return self.dimensions.value