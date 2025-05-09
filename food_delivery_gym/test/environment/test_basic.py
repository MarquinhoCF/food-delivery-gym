import unittest

from food_delivery_gym.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv
from food_delivery_gym.main.generator.time_shift_customer_generator import TimeShiftCustomerGenerator
from food_delivery_gym.main.generator.time_shift_driver_generator import TimeShiftDriverGenerator
from food_delivery_gym.main.generator.time_shift_order_generator import TimeShiftOrderGenerator
from food_delivery_gym.main.generator.time_shift_establishment_generator import TimeShiftEstablishmentGenerator
from food_delivery_gym.main.map.grid_map import GridMap
from food_delivery_gym.main.optimizer.random_driver_optimizer import RandomDriverOptimizer


class TestBasic(unittest.TestCase):

    def runTest(self):
        environment = FoodDeliverySimpyEnv(
            map=GridMap(100),
            generators=[
                TimeShiftCustomerGenerator(lambda time: 3),
                TimeShiftEstablishmentGenerator(lambda time: 3),
                TimeShiftDriverGenerator(lambda time: 10),
                TimeShiftOrderGenerator(lambda time: 2 * time)
            ],
            optimizer=RandomDriverOptimizer()
        )

        self.assertEqual(environment.state.drivers, [])

        environment.run(until=2)

        self.assertTrue(len(environment.events) > 0)
