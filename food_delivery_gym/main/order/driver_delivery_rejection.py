from simpy.core import SimTime

from food_delivery_gym.main.order.delivery_rejection import DeliveryRejection
from food_delivery_gym.main.order.delivery_rejection_type import DeliveryRejectionType


class DriverDeliveryRejection(DeliveryRejection):
    def __init__(self, driver, time: SimTime):
        super().__init__(time, DeliveryRejectionType.REJECTED_BY_DRIVER)
        self.driver = driver
