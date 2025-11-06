from enum import Enum, auto


class DriverStatus(Enum):
    AVAILABLE = auto()
    PROCESSING_PREVIOUS_ORDERS = auto()
    PICKING_UP = auto()
    PICKING_UP_WAITING = auto()
    DELIVERING = auto()
    DELIVERING_WAITING = auto()
