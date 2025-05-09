from food_delivery_gym.main.base.dimensions import Dimensions


class Item:
    def __init__(self, item_type, dimensions: Dimensions, preparation_time):
        self.item_type = item_type
        self.dimensions = dimensions
        self.preparation_time = preparation_time
