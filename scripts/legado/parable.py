from food_delivery_gym.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv
from food_delivery_gym.main.generator.time_shift_customer_generator import TimeShiftCustomerGenerator
from food_delivery_gym.main.generator.time_shift_driver_generator import TimeShiftDriverGenerator
from food_delivery_gym.main.generator.time_shift_order_generator import TimeShiftOrderGenerator
from food_delivery_gym.main.generator.time_shift_establishment_generator import TimeShiftEstablishmentGenerator
from food_delivery_gym.main.map.grid_map import GridMap
from food_delivery_gym.main.optimizer.random_driver_optimizer import RandomDriverOptimizer
from food_delivery_gym.main.statistic.custom_board import CustomBoard
from food_delivery_gym.main.statistic.delay_metric import DelayMetric
from food_delivery_gym.main.statistic.distance_metric import DistanceMetric
from food_delivery_gym.main.statistic.driver_status_metric import DriverStatusMetric
from food_delivery_gym.main.statistic.order_curve_metric import OrderCurveMetric
from food_delivery_gym.main.statistic.order_status_metric import OrderStatusMetric
from food_delivery_gym.main.statistic.total_metric import TotalMetric

a = -4/225 # Largura e direção
b = 250 # Ponto de inflexão
c = 400 # Pico da parabola


def parable(time):
    return max(2, int(a * pow(time - b, 2) + c))


def run():
    environment = FoodDeliverySimpyEnv(
        map=GridMap(100),
        generators=[
            TimeShiftCustomerGenerator(lambda time: 3),
            TimeShiftEstablishmentGenerator(lambda time: 3, use_estimate=True),
            TimeShiftDriverGenerator(lambda time: 3),
            TimeShiftOrderGenerator(lambda time: parable(time))
        ],
        optimizer=RandomDriverOptimizer()
    )
    environment.run(until=2000)

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
