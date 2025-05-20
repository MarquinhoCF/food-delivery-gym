from enum import Enum, auto

class OrderStatus(Enum):
    CREATED = auto()
    PLACED = auto()
    
    ESTABLISHMENT_ACCEPTED = auto()
    ESTABLISHMENT_REJECTED = auto()
    
    PREPARING = auto()
    READY = auto()
    
    DRIVER_REJECTED = auto()
    READY_AND_DRIVER_REJECTED = auto()
    PREPARING_AND_DRIVER_REJECTED = auto()

    DRIVER_ACCEPTED = auto()
    READY_AND_DRIVER_ACCEPTED = auto()
    PREPARING_AND_DRIVER_ACCEPTED = auto()

    PICKING_UP = auto()
    PREPARING_AND_PICKING_UP = auto()
    READY_AND_PICKING_UP = auto()

    PICKED_UP = auto()
    DELIVERING = auto()
    DRIVER_ARRIVED_DELIVERY_LOCATION = auto()
    RECEIVED = auto()
    DELIVERED = auto()

    def __lt__(self, other):
        return self.value < other.value

    def __gt__(self, other):
        return self.value > other.value

    def __le__(self, other):
        return self.value <= other.value

    def __ge__(self, other):
        return self.value >= other.value
