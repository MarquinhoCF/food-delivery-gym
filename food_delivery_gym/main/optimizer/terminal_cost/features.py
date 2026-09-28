"""
Features de estado para o custo terminal do rollout.

A mesma função é usada na coleta (episódios da política de base) e na
inferência (`terminal_cost_to_go` sobre o ambiente clonado), então ela só
lê informação disponível no instante da decisão.
"""

import numpy as np

from food_delivery_gym.main.driver.driver_status import DriverStatus

FEATURE_NAMES: tuple[str, ...] = (
    "time_fraction",
    "orders_delivered",
    "orders_open",
    "orders_generated",
    "busy_time_sum",
    "busy_time_mean",
    "busy_time_max",
    "queue_size_sum",
    "queue_size_mean",
    "queue_size_max",
    "free_drivers_fraction",
    "order_ready_time",
    "order_delivery_time_mean",
)


def extract_features(env) -> np.ndarray:
    """Extrai o vetor de features (float64, shape=(len(FEATURE_NAMES),)) do estado atual."""
    obs = env.get_observation()
    simpy_env = env.simpy_env

    busy = np.asarray(obs["drivers_estimated_remaining_time"], dtype=np.float64)
    queue = np.asarray(obs["drivers_queue_size"], dtype=np.float64)
    status = np.asarray(obs["driver_status"], dtype=np.float64)
    delivery_times = np.asarray(obs["order_estimated_delivery_time"], dtype=np.float64)

    orders_delivered = float(simpy_env.state.get_orders_delivered())
    orders_generated = float(env.orders_generated or 0)
    orders_open = max(0.0, float(len(simpy_env.state.orders)) - orders_delivered)

    return np.array(
        [
            float(simpy_env.now) / float(env.max_time_step),
            orders_delivered,
            orders_open,
            orders_generated,
            float(busy.sum()),
            float(busy.mean()),
            float(busy.max()),
            float(queue.sum()),
            float(queue.mean()),
            float(queue.max()),
            float(np.mean(status == float(DriverStatus.AVAILABLE.value))),
            float(obs["order_estimated_ready_time"][0]),
            float(delivery_times.mean()),
        ],
        dtype=np.float64,
    )
