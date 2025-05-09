from abc import ABC, abstractmethod


class FoodDeliveryView(ABC):

    def __init__(self, grid_size=100, window_size=(800, 600), fps=30):
        self.grid_size = grid_size
        self.window_size = window_size
        self.fps = fps
        self.min_x = 0
        self.max_x = grid_size
        self.min_y = 0
        self.max_y = grid_size
        self.quited = False

    @abstractmethod
    def render(self, environment) -> bool: pass

    @abstractmethod
    def quit(self): pass
