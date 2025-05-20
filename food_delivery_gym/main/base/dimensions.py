from food_delivery_gym.main.base.types import Number


class Dimensions:
    def __init__(self, length: Number, height: Number, width: Number, weight: Number):
        self.length = length
        self.height = height
        self.width = width
        self.weight = weight

    def __lt__(self, other: 'Dimensions'):
        return (
                self.length < other.length and
                self.height < other.height and
                self.width < other.width and
                self.weight < other.weight
        )

    def __gt__(self, other: 'Dimensions'):
        return (
                self.length > other.length and
                self.height > other.height and
                self.width > other.width and
                self.weight > other.weight
        )

    def __eq__(self, other: 'Dimensions'):
        return (
                self.length == other.length and
                self.height == other.height and
                self.width == other.width and
                self.weight == other.weight
        )

    def __add__(self, other: 'Dimensions'):
        return Dimensions(
            self.length + other.length,
            self.height + other.height,
            self.width + other.width,
            self.weight + other.weight
        )

    @property
    def volume(self) -> Number:
        return self.length * self.height * self.width

    @property
    def value(self) -> Number:
        return self.volume * self.weight
