import random
import numpy as np

class RandomManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RandomManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.seed = None
        self._random_instance = np.random.default_rng() 

    def set_seed(self, seed: int | None = None):
        if seed is not None:
            self.seed = seed
            self._random_instance = np.random.default_rng(seed)

    def get_random_instance(self):
        return self._random_instance
