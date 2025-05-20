from enum import Enum, auto

class CustumerStatus(Enum):
    WAITING_DELIVERY = auto()
    DELIVERED = auto()