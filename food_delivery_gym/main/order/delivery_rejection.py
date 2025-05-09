from simpy.core import SimTime

from food_delivery_gym.main.order.delivery_rejection_type import DeliveryRejectionType


class DeliveryRejection:
    def __init__(self, time: SimTime, rejection_type: DeliveryRejectionType):
        self.time = time
        self.rejection_type = rejection_type
