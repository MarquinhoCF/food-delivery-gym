from food_delivery_gym.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv
from food_delivery_gym.main.generator.initial_customer_generator import InitialCustomerGenerator
from food_delivery_gym.main.generator.initial_driver_generator import InitialDriverGenerator
from food_delivery_gym.main.generator.initial_order_generator import InitialOrderGenerator
from food_delivery_gym.main.generator.initial_establishment_generator import InitialEstablishmentGenerator
from food_delivery_gym.main.map.grid_map import GridMap
from food_delivery_gym.main.optimizer.random_driver_optimizer import RandomDriverOptimizer
from food_delivery_gym.main.view.grid_view_matplotlib import GridViewMatplotlib


def main():
    environment = FoodDeliverySimpyEnv(
        map=GridMap(100),
        generators=[
            InitialCustomerGenerator(10),
            InitialEstablishmentGenerator(10),
            InitialDriverGenerator(10),
            InitialOrderGenerator(10)
        ],
        optimizer=RandomDriverOptimizer(),
        view=GridViewMatplotlib()
    )
    environment.run(100, render_mode='human')


if __name__ == '__main__':
    main()
