from food_delivery_gym.main.cost.simple_cost_function import SimpleCostFunction
from food_delivery_gym.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv
from food_delivery_gym.main.generator.initial_driver_generator import InitialDriverGenerator
from food_delivery_gym.main.generator.initial_establishment_order_rate_generator import InitialEstablishmentOrderRateGenerator
from food_delivery_gym.main.generator.time_shift_order_establishment_rate_generator import TimeShiftOrderEstablishmentRateGenerator
from food_delivery_gym.main.map.grid_map import GridMap
from food_delivery_gym.main.optimizer.or_tools_optimizer import OrToolsOptimizer
from food_delivery_gym.main.statistic.custom_board import CustomBoard
from food_delivery_gym.main.statistic.delay_metric import DelayMetric
from food_delivery_gym.main.statistic.distance_metric import DistanceMetric
from food_delivery_gym.main.statistic.driver_status_metric import DriverStatusMetric
from food_delivery_gym.main.statistic.order_curve_metric import OrderCurveMetric
from food_delivery_gym.main.statistic.order_status_metric import OrderStatusMetric
from food_delivery_gym.main.statistic.total_metric import TotalMetric
from food_delivery_gym.main.view.grid_view_pygame import GridViewPygame


def run():
    environment = FoodDeliverySimpyEnv(
        map=GridMap(100),
        generators=[
            InitialEstablishmentOrderRateGenerator(10),
            InitialDriverGenerator(10),
            TimeShiftOrderEstablishmentRateGenerator(lambda time: 1 if time < 400 else 0),
        ],
        optimizer=OrToolsOptimizer(cost_function=SimpleCostFunction()),
        view=GridViewPygame()
    )
    environment.run(500, render_mode='human')

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
