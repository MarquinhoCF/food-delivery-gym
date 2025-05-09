from enum import Enum, auto


class DeliveryRejectionType(Enum):
    REJECTED_BY_DRIVER = auto()
    REJECTED_BY_OPTIMIZATION = auto()
