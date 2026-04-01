from food_delivery_gym.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv
from food_delivery_gym.main.generator.initial_driver_generator import InitialDriverGenerator
from food_delivery_gym.main.generator.initial_establishment_order_rate_generator import InitialEstablishmentOrderRateGenerator
from food_delivery_gym.main.generator.time_shift_order_establishment_rate_generator import TimeShiftOrderEstablishmentRateGenerator
from food_delivery_gym.main.map.grid_map import GridMap
from food_delivery_gym.main.optimizer.optimizer_simpy.random_driver_optimizer_simpy import RandomDriverOptimizerSimpy
from food_delivery_gym.main.statistic.custom_board import CustomBoard
from food_delivery_gym.main.statistic.delay_metric import DelayMetric
from food_delivery_gym.main.statistic.distance_metric import DistanceMetric
from food_delivery_gym.main.statistic.driver_status_metric import DriverStatusMetric
from food_delivery_gym.main.statistic.order_curve_metric import OrderCurveMetric
from food_delivery_gym.main.statistic.order_status_metric import OrderStatusMetric
from food_delivery_gym.main.statistic.total_metric import TotalMetric


def run():
    environment = FoodDeliverySimpyEnv(
        map=GridMap(100),
        generators=[
            InitialEstablishmentOrderRateGenerator(100, use_estimate=True),
            InitialDriverGenerator(20),
            TimeShiftOrderEstablishmentRateGenerator(lambda time: 1),
        ],
        optimizer=RandomDriverOptimizerSimpy()
    )
    environment.run(100, render_mode='human')

    custom_board = CustomBoard(metrics=[
        OrderCurveMetric(environment),
        TotalMetric(environment),
        DistanceMetric(environment),
        DelayMetric(environment),
        DriverStatusMetric(environment),
        OrderStatusMetric(environment),
    ])
    custom_board.view()


if __name__ == '__main__':
    run()
