from food_delivery_gym.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv
from food_delivery_gym.main.generator.time_shift_customer_generator import TimeShiftCustomerGenerator
from food_delivery_gym.main.generator.time_shift_driver_generator import TimeShiftDriverGenerator
from food_delivery_gym.main.generator.time_shift_order_generator import TimeShiftOrderGenerator
from food_delivery_gym.main.generator.time_shift_establishment_generator import TimeShiftEstablishmentGenerator
from food_delivery_gym.main.map.grid_map import GridMap
from food_delivery_gym.main.optimizer.nearest_driver_optimizer import NearestDriverOptimizer
from food_delivery_gym.main.statistic.default_board import DefaultBoard
from food_delivery_gym.main.statistic.delay_metric import DelayMetric
from food_delivery_gym.main.statistic.distance_metric import DistanceMetric
from food_delivery_gym.main.statistic.driver_status_metric import DriverStatusMetric
from food_delivery_gym.main.statistic.order_curve_metric import OrderCurveMetric
from food_delivery_gym.main.statistic.order_status_metric import OrderStatusMetric
from food_delivery_gym.main.statistic.total_metric import TotalMetric


def main():
    environment = FoodDeliverySimpyEnv(
        map=GridMap(100),
        generators=[
            TimeShiftCustomerGenerator(lambda time: 3),
            TimeShiftEstablishmentGenerator(lambda time: 3),
            TimeShiftDriverGenerator(lambda time: 10),
            TimeShiftOrderGenerator(lambda time: time * 2 if time <= 100 else 1)
        ],
        optimizer=NearestDriverOptimizer(),
        # view=GridViewMatplotlib()
    )
    environment.run(until=200)
    # environment.log_events()

    custom_board = DefaultBoard(metrics=[
        OrderCurveMetric(environment),
        TotalMetric(environment),
        DistanceMetric(environment),
        DelayMetric(environment),
        DriverStatusMetric(environment),
        OrderStatusMetric(environment),
    ])
    custom_board.view()


if __name__ == '__main__':
    main()
