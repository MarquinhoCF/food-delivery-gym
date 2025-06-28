import numpy as np

from food_delivery_gym.main.statistic.driver_idle_time_metric import DriverIdleTimeMetric
from food_delivery_gym.main.statistic.driver_orders_delivered_metric import DriverOrdersDeliveredMetric
from food_delivery_gym.main.statistic.driver_time_waiting_for_order_metric import DriverTimeWaitingForOrderMetric
from food_delivery_gym.main.statistic.driver_total_distance_metric import DriverTotalDistanceMetric
from food_delivery_gym.main.statistic.establishment_active_time_metric import EstablishmentActiveTimeMetric
from food_delivery_gym.main.statistic.establishment_idle_time_metric import EstablishmentIdleTimeMetric
from food_delivery_gym.main.statistic.establishment_max_orders_in_queue_metric import EstablishmentMaxOrdersInQueueMetric
from food_delivery_gym.main.statistic.establishment_orders_fulfilled_metric import EstablishmentOrdersFulfilledMetric
from food_delivery_gym.main.statistic.summarized_data_board import SummarizedDataBoard


DIR_PATH = "./data/test/random_heuristic/"
NUM_DRIVERS = 10
NUM_ESTABLISHMENTS = 10

def main():
    dados = np.load(DIR_PATH + "metrics_Data.npz", allow_pickle=True)
    
    total_rewards = dados["total_rewards"]

    T_1 = dados["total_rewards_statistics"]  # Isso retorna um array do NumPy
    total_rewards_statistics = None
    if isinstance(T_1, np.ndarray) and T_1.dtype == object:
        total_rewards_statistics = T_1.item() 

    T_2 = dados["establishment_metrics"]  # Isso retorna um array do NumPy
    establishment_metrics = None
    if isinstance(T_1, np.ndarray) and T_2.dtype == object:
        establishment_metrics = T_2.item()  # Converte o array do NumPy em um dicionário Python

    T_3 = dados["driver_metrics"]  # Isso retorna um array do NumPy
    driver_metrics = None
    if isinstance(T_3, np.ndarray) and T_3.dtype == object:
        driver_metrics = T_3.item()  # Converte o array do NumPy em um dicionário Python

    T_4 = dados["geral_statistics"]  # Isso retorna um array do NumPy
    geral_statistics = None
    if isinstance(T_4, np.ndarray) and T_4.dtype == object:
        geral_statistics = T_4.item()  # Converte o array do NumPy em um dicionário Python

    # Calculando a média dos valores de recompensa
    sum_rewards_mean = total_rewards_statistics["avg"]

    custom_board = SummarizedDataBoard(metrics=[
        EstablishmentOrdersFulfilledMetric(None, establishments_statistics=geral_statistics["establishments"]),
        EstablishmentMaxOrdersInQueueMetric(None, establishments_statistics=geral_statistics["establishments"]),
        EstablishmentActiveTimeMetric(None, establishments_statistics=geral_statistics["establishments"]),
        EstablishmentIdleTimeMetric(None, establishments_statistics=geral_statistics["establishments"]),
        DriverOrdersDeliveredMetric(None, drivers_statistics=geral_statistics["drivers"]),
        DriverTotalDistanceMetric(None, drivers_statistics=geral_statistics["drivers"]),
        DriverIdleTimeMetric(None, drivers_statistics=geral_statistics["drivers"]),
        DriverTimeWaitingForOrderMetric(None, drivers_statistics=geral_statistics["drivers"])
    ],
        num_drivers=NUM_DRIVERS,
        num_establishments=NUM_ESTABLISHMENTS,
        sum_reward=sum_rewards_mean,
        save_figs=True,
        dir_path=DIR_PATH,
        use_total_mean=True,
        use_tkinter=False
    )
    custom_board.view()


if __name__ == '__main__':
    main()
