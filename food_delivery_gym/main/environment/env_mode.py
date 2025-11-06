from enum import Enum, auto

class EnvMode(Enum):
    TESTING = auto()
    TRAINING = auto()
    EVALUATING = auto()

    def __lt__(self, other):
        return self.value < other.value

    def __gt__(self, other):
        return self.value > other.value

    def __le__(self, other):
        return self.value <= other.value

    def __ge__(self, other):
        return self.value >= other.value
