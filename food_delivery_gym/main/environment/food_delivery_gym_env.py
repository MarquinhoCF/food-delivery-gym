import json
from pathlib import Path
import traceback
from typing import Optional
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Dict, Box, Discrete

from food_delivery_gym.main.environment.env_mode import EnvMode
from food_delivery_gym.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv
from food_delivery_gym.main.generator.initial_driver_generator import InitialDriverGenerator
from food_delivery_gym.main.generator.initial_establishment_order_rate_generator import InitialEstablishmentOrderRateGenerator
from food_delivery_gym.main.generator.time_shift_order_establishment_rate_generator import TimeShiftOrderEstablishmentRateGenerator
from food_delivery_gym.main.map.grid_map import GridMap
from food_delivery_gym.main.route.delivery_route_segment import DeliveryRouteSegment
from food_delivery_gym.main.route.pickup_route_segment import PickupRouteSegment
from food_delivery_gym.main.route.route import Route
from food_delivery_gym.main.statistic.driver_idle_time_metric import DriverIdleTimeMetric
from food_delivery_gym.main.statistic.driver_time_waiting_for_order_metric import DriverTimeWaitingForOrderMetric
from food_delivery_gym.main.statistic.summarized_data_board import SummarizedDataBoard
from food_delivery_gym.main.statistic.driver_orders_delivered_metric import DriverOrdersDeliveredMetric
from food_delivery_gym.main.statistic.driver_total_distance_metric import DriverTotalDistanceMetric
from food_delivery_gym.main.statistic.establishment_active_time_metric import EstablishmentActiveTimeMetric
from food_delivery_gym.main.statistic.establishment_idle_time_metric import EstablishmentIdleTimeMetric
from food_delivery_gym.main.statistic.establishment_max_orders_in_queue_metric import EstablishmentMaxOrdersInQueueMetric
from food_delivery_gym.main.statistic.establishment_orders_fulfilled_metric import EstablishmentOrdersFulfilledMetric
from food_delivery_gym.main.statistic.order_curve_metric import OrderCurveMetric
from food_delivery_gym.main.utils.random_manager import RandomManager
from food_delivery_gym.main.view.grid_view_pygame import GridViewPygame

class FoodDeliveryGymEnv(Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, scenario_json_file_path = str):

        path = Path(scenario_json_file_path)

        if not path.is_file():
            raise FileNotFoundError(f"Scenario file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            scenario = json.load(f)

        required_keys = [
            "num_drivers",
            "num_establishments",
            "num_orders",
            "num_costumers",
            "function_code",
            "time_shift",
            "vel_drivers",
            "prepare_time",
            "operating_radius",
            "production_capacity",
            "percentage_allocation_driver",
            "max_time_step",
            "reward_objective"
        ]

        missing_keys = [key for key in required_keys if key not in scenario]
        if missing_keys:
            raise ValueError(f"Missing required keys in scenario_json: {missing_keys}")

        self.num_drivers = scenario["num_drivers"]
        self.num_establishments = scenario["num_establishments"]
        self.num_orders = scenario["num_orders"]
        self.num_costumers = scenario["num_costumers"]
        self.function = eval(scenario["function_code"])
        self.lambda_code = scenario["function_code"]
        self.time_shift = scenario["time_shift"]
        self.vel_drivers = scenario["vel_drivers"]
        self.prepare_time = scenario["prepare_time"]
        self.operating_radius = scenario["operating_radius"]
        self.production_capacity = scenario["production_capacity"]
        self.percentage_allocation_driver = scenario["percentage_allocation_driver"]
        self.grid_map_size = scenario.get("grid_map_size", 100)
        self.use_estimate = scenario.get("use_estimate", True)
        self.desconsider_capacity = scenario.get("desconsider_capacity", True)
        self.max_time_step = scenario["max_time_step"]
        self.normalize = scenario.get("normalize", True)
        self.env_mode = EnvMode.TRAINING
        
        render_mode = scenario.get("render_mode", None)
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.simpy_env = None # Ambiente de simulação será criado no reset

        # Definindo o objetivo da recompensa
        self.set_reward_objective(scenario["reward_objective"])

        # Espaço de Observação
        if self.normalize:
            self.dtype_observation = np.float32

            self.observation_space = Dict({
                'drivers_busy_time': Box(low=-1, high=1, shape=(self.num_drivers,), dtype=self.dtype_observation),
                'time_to_drivers_complete_order': Box(low=-1, high=1, shape=(self.num_drivers,), dtype=self.dtype_observation),
                'remaining_orders': Box(low=-1, high=1, shape=(1,), dtype=self.dtype_observation),
                'establishment_busy_time': Box(low=-1, high=1, shape=(self.num_establishments,), dtype=self.dtype_observation),
                'current_time_step': Box(low=-1, high=1, shape=(1,), dtype=self.dtype_observation)
            })
            
            self.limits_observation_space = {
                'drivers_busy_time': (0, self.max_time_step),
                'time_to_drivers_complete_order': (0, self.max_time_step),
                'remaining_orders': (0, self.num_orders + 1),
                'establishment_busy_time': (0, self.max_time_step),
                'current_time_step': (0, self.max_time_step)
            }
        else:
            self.dtype_observation = np.int32
            self.observation_space = Dict({
                'drivers_busy_time': Box(low=0, high=self.max_time_step, shape=(self.num_drivers,), dtype=self.dtype_observation),
                'time_to_drivers_complete_order': Box(low=0, high=self.max_time_step, shape=(self.num_drivers,), dtype=self.dtype_observation),
                'remaining_orders': Discrete(self.num_orders + 1),
                'establishment_busy_time': Box(low=0, high=self.max_time_step, shape=(self.num_establishments,), dtype=self.dtype_observation),
                'current_time_step': Discrete(self.max_time_step)
            })

        # Espaço de Ação
        self.action_space = Discrete(self.num_drivers)  # Escolher qual driver pegará o pedido

    def normalize_observation(self, obs):
        normalized_obs = {}
        for key, value in obs.items():
            min_value, max_value = self.limits_observation_space[key]
            normalized_value = 2 * (value - min_value) / (max_value - min_value) - 1
        
            # Garantir que o valor esteja dentro do intervalo [-1, 1]
            normalized_obs[key] = np.clip(normalized_value, -1, 1)

        return normalized_obs

    def get_observation(self):
        # 1. drivers_busy_time: Tempo de ocupação de cada motorista
        drivers_busy_time = np.zeros((self.num_drivers,), dtype=self.dtype_observation)
        for i, driver in enumerate(self.simpy_env.state.drivers):
            drivers_busy_time[i] = driver.estimate_total_busy_time()

        # 2. time_to_drivers_complete_order: Tempo estimado para cada motorista completar o próximo pedido
        time_to_drivers_complete_order = np.zeros((self.num_drivers,), dtype=self.dtype_observation)
        for i, driver in enumerate(self.simpy_env.state.drivers):
            time_to_drivers_complete_order[i] = driver.estimate_time_to_complete_next_order(self.last_order)

        # 3. orders_remaining: Número de pedidos que faltam ser atribuidos a um motorista
        orders_remaining = self.num_orders - self.simpy_env.state.successfully_assigned_routes
        orders_remaining = np.array([orders_remaining], dtype=self.dtype_observation) if self.normalize else orders_remaining

        # 4. establishment_next_order_ready_time: Tempo que falta para o próximo pedido em preparação de cada restaurante ficar pronto
        establishment_busy_time = np.zeros((self.num_establishments,), dtype=self.dtype_observation)
        for i, establishment in enumerate(self.simpy_env.state.establishments):
            establishment_busy_time[i] = establishment.calculate_mean_overload_time()

        # 5. current_time_step: O tempo atual da simulação (número do passo)
        current_time_step = self.simpy_env.now
        current_time_step = np.array([current_time_step], dtype=self.dtype_observation) if self.normalize else current_time_step

        # Criando a observação final no formato esperado
        obs = {
            'drivers_busy_time': drivers_busy_time,
            'time_to_drivers_complete_order': time_to_drivers_complete_order,
            'remaining_orders': orders_remaining,
            'establishment_busy_time': establishment_busy_time,
            'current_time_step': current_time_step
        }

        return self.normalize_observation(obs) if self.normalize else obs
       
    def _get_info(self):
        return {'info': self.simpy_env.now}
    
    def set_mode(self, mode: EnvMode):
        self.env_mode = mode
    
    # Avança na simulação até que um evento principal ocorra ou que a simulação termine/trunque.
    def advance_simulation_until_event(self):
        terminated = False
        truncated = False
        core_event = None
        
        while (not terminated) and (not truncated) and (core_event is None):
            if self.simpy_env.state.orders_delivered < self.num_orders:
                self.simpy_env.step(self.env_mode, self.render_mode)
                
                # TODO: Logs
                # # Verifica se um pedido foi entregue
                # if self.simpy_env.state.orders_delivered > self.last_num_orders_delivered:
                    # print("Pedido entregue!")
                    # print(f"Número de pedidos entregues: {self.simpy_env.state.orders_delivered}")
                    # self.last_num_orders_delivered = self.simpy_env.state.orders_delivered

                # Verifica o próximo evento principal
                core_event = self.simpy_env.dequeue_core_event()

                # TODO: Logs
                # if core_event is not None:
                #     print('\n----> Pedido atual para alocação do motorista <----')
                #     print(core_event.order)
                
                # Verifica se atingiu o limite de tempo
                if self.simpy_env.now >= self.max_time_step - 1:
                    # print("Limite de tempo atingido!")
                    truncated = True
            else:
                # TODO: Logs
                # print("Todos os pedidos foram entregues!")
                terminated = True

        return core_event, terminated, truncated

    def reset(self, seed: int | None = None, options: Optional[dict] = None):
        if seed is not None:
            super().reset(seed=seed)
            self.action_space.seed(seed=seed)
            RandomManager().set_seed(seed=seed)

        if options:
            self.render_mode = options.get("render_mode", None)

        self.simpy_env = FoodDeliverySimpyEnv(
            map=GridMap(self.grid_map_size),
            generators=[
                InitialEstablishmentOrderRateGenerator(
                    self.num_establishments, 
                    self.prepare_time, 
                    self.operating_radius, 
                    self.production_capacity,
                    self.percentage_allocation_driver, 
                    use_estimate=self.use_estimate,
                ),
                InitialDriverGenerator(
                    self.num_drivers, 
                    self.vel_drivers, 
                    self.reward_objective,
                    desconsider_capacity=self.desconsider_capacity,
                ),
                TimeShiftOrderEstablishmentRateGenerator(
                    self.function, 
                    time_shift=self.time_shift, 
                    max_orders=self.num_orders,
                ),
            ],
            optimizer=None,
            view=GridViewPygame(grid_size=self.grid_map_size) if self.render_mode == "human" else None
        )

        # self.last_num_orders_delivered = 0
        core_event, _, _ = self.advance_simulation_until_event()
        self.last_order = core_event.order if core_event else None

        observation = self.get_observation()
        info = self._get_info()

        return observation, info
    
    def select_driver_to_order(self, selected_driver, order):
        segment_pickup = PickupRouteSegment(order)
        segment_delivery = DeliveryRouteSegment(order)
        route = Route(self.simpy_env, [segment_pickup, segment_delivery])
        selected_driver.receive_route_requests(route)

    def calculate_reward(self, terminated, truncated):
        reward = 0

        # Objetivo 1: Minimizar o tempo de entrega a partir da expectativa de tempo gasto com a entrega-> Recompensa negativa a cada passo
        if self.reward_objective == 1:
            # Soma das estimativas do tempo de ocupação de cada motoristas
            reward = -sum(driver.estimate_total_busy_time() for driver in self.simpy_env.state.drivers)

        # Objetivo 2: Minimizar o tempo de entrega a partir da expectativa de tempo gasto com a entrega -> Recompensa negativa ao final do episódio
        elif self.reward_objective == 2:
            # Atualiza as estimativas de tempo de ocupação de cada motorista
            for driver in self.simpy_env.state.drivers:
                driver.update_expected_delivery_time_reward()

            if terminated or truncated:
                # Penalidade baseada na soma das estimativas de tempo de ocupação dos motoristas
                reward = -sum(driver.get_expected_delivery_time_reward() for driver in self.simpy_env.state.drivers)
            
        # Objetivo 3: Minimizar o tempo de entrega dos motoristas a partir do tempo efetivo gasto -> Recompensa negativa a cada passo
        # Objetivo 5: Minimizar o tempo de entrega dos motoristas a partir do tempo efetivo gasto (Com penalização 5x para pedidos não coletados) -> Recompensa negativa a cada passo
        elif self.reward_objective in [3, 5]:
            # Soma do tempo efetivo gasto por cada motorista
            reward = -sum(driver.get_penality_for_time_spent_for_delivery() for driver in self.simpy_env.state.drivers)

            if truncated and self.simpy_env.state.orders_delivered < self.num_orders:
                # Se a simulação foi truncada e não foram entregues todos os pedidos, penaliza a recompensa
                reward -= sum(driver.get_penality_for_late_orders() for driver in self.simpy_env.state.drivers)

        # Objetivo 4: Minimizar o tempo de entrega dos motoristas a partir do tempo efetivo gasto -> Recompensa negativa no fim do episódio
        # Objetivo 6: Minimizar o tempo de entrega dos motoristas a partir do tempo efetivo gasto (Com penalização 5x para pedidos não coletados) -> Recompensa negativa no fim do episódio
        elif self.reward_objective in [4, 6] and (terminated or truncated):
            # Soma do tempo efetivo gasto por cada motorista
            reward = -sum(driver.get_penality_for_time_spent_for_delivery() for driver in self.simpy_env.state.drivers)

            if truncated and self.simpy_env.state.orders_delivered < self.num_orders:
                # Se a simulação foi truncada e não foram entregues todos os pedidos, penaliza a recompensa
                reward -= sum(driver.get_penality_for_late_orders() for driver in self.simpy_env.state.drivers)

        # Objetivo 7: Minimizar o custo de operação a partir da distância efetiva -> Recompensa negativa a cada passo
        elif self.reward_objective == 7:
            # Distância percorrida desde a última recompensa para o motorista selecionado
            reward = -sum(driver.get_and_update_distance_traveled() for driver in self.simpy_env.state.drivers)

            if truncated and self.simpy_env.state.orders_delivered < self.num_orders:
                # Se a simulação foi truncada e não foram entregues todos os pedidos, penaliza a recompensa
                reward -= (self.num_orders - self.simpy_env.state.orders_delivered) * self.simpy_env.map.max_distance() * 2

        # Objetivo 8: Minimizar o custo de operação a partir da distância efetiva -> Recompensa negativa no fim do episódio
        elif self.reward_objective == 8 and (terminated or truncated):
            # Distância total percorrida por cada motorista
            reward = -sum(driver.get_and_update_distance_traveled() for driver in self.simpy_env.state.drivers)

            if truncated and self.simpy_env.state.orders_delivered < self.num_orders:
                # Se a simulação foi truncada e não foram entregues todos os pedidos, penaliza a recompensa
                reward -= (self.num_orders - self.simpy_env.state.orders_delivered) * self.simpy_env.map.max_distance() * 2

        # Objetivo 9: Minimizar o custo de operação a partir da distância a ser percorrida -> Recompensa negativa a cada passo
        if self.reward_objective == 9:
            # Soma das estimativas do tempo de ocupação de cada motoristas
            reward = -sum(driver.calculate_total_distance_to_travel() for driver in self.simpy_env.state.drivers)

        # Objetivo 10: Minimizar o custo de operação a partir da distância a ser percorrida -> Recompensa negativa ao final do episódio
        elif self.reward_objective == 10:
            # Atualiza as estimativas de tempo de ocupação de cada motorista
            for driver in self.simpy_env.state.drivers:
                driver.update_distance_to_be_traveled_reward()

            if terminated or truncated:
                # Penalidade baseada na soma das estimativas de tempo de ocupação dos motoristas
                reward = -sum(driver.get_distance_to_be_traveled_reward() for driver in self.simpy_env.state.drivers)
        
        return reward
        
    def step(self, action):
        try:
            if action < 0 or action >= self.num_drivers:
                raise ValueError(f"A ação {action} é inválida! Deve ser um número entre 0 e {self.num_drivers}")

            if self.render_mode == "human":
                truncated = self.simpy_env.view.quited

            terminated = False
            # print("action: {}".format(action))
            # print("last_order: {}".format(vars(self.last_order)))
            selected_driver = self.simpy_env.state.drivers[action]
            self.select_driver_to_order(selected_driver, self.last_order)

            core_event, terminated, truncated = self.advance_simulation_until_event()

            self.last_order = core_event.order if core_event else None

            observation = self.get_observation()

            # assert self.observation_space.contains(observation), "A observação gerada não está contida no espaço de observação."
            
            info = self._get_info()

            reward = self.calculate_reward(terminated, truncated)
            # print(f"reward: {reward}")

            return observation, reward, terminated, truncated, info
        
        except ValueError as e:
            print(e)
            print(traceback.format_exc())
            raise

        except AttributeError as e:
            print("Erro ao executar o passo da simulação!\nVerifique se o método reset() foi chamado antes de utilizar o ambiente.")
            print(traceback.format_exc())
            raise

        except Exception as e:
            print(e)
            print(traceback.format_exc())
            raise

    def show_statistcs_board(self, sum_reward = None, dir_path = None):
        if sum_reward is None and dir_path is None:
            save_figs = False
        else:
            save_figs = True
        custom_board = SummarizedDataBoard(metrics=[
            OrderCurveMetric(self.simpy_env),
            EstablishmentOrdersFulfilledMetric(self.simpy_env),
            EstablishmentMaxOrdersInQueueMetric(self.simpy_env),
            EstablishmentActiveTimeMetric(self.simpy_env),
            EstablishmentIdleTimeMetric(self.simpy_env),
            DriverOrdersDeliveredMetric(self.simpy_env),
            DriverTotalDistanceMetric(self.simpy_env),
            DriverIdleTimeMetric(self.simpy_env),
            DriverTimeWaitingForOrderMetric(self.simpy_env)
        ],
            num_drivers=self.num_drivers,
            num_establishments=self.num_establishments,
            sum_reward=sum_reward,
            save_figs=save_figs,
            dir_path=dir_path,
            use_total_mean=False,
            use_tkinter=False
        )
        custom_board.view()
    
    def show_total_mean_statistics_board(self, sum_rewards_mean = None, dir_path= None):
        if sum_rewards_mean is None and dir_path is None:
            save_figs = False
        else:
            save_figs = True
        
        statistics = self.get_statistics()

        custom_board = SummarizedDataBoard(metrics=[
            EstablishmentOrdersFulfilledMetric(self.simpy_env, establishments_statistics=statistics["establishments"]),
            EstablishmentMaxOrdersInQueueMetric(self.simpy_env, establishments_statistics=statistics["establishments"]),
            EstablishmentActiveTimeMetric(self.simpy_env, establishments_statistics=statistics["establishments"]),
            EstablishmentIdleTimeMetric(self.simpy_env, establishments_statistics=statistics["establishments"]),
            DriverOrdersDeliveredMetric(self.simpy_env, drivers_statistics=statistics["drivers"]),
            DriverTotalDistanceMetric(self.simpy_env, drivers_statistics=statistics["drivers"]),
            DriverIdleTimeMetric(self.simpy_env, drivers_statistics=statistics["drivers"]),
            DriverTimeWaitingForOrderMetric(self.simpy_env, drivers_statistics=statistics["drivers"])
        ],
            num_drivers=self.num_drivers,
            num_establishments=self.num_establishments,
            sum_reward=sum_rewards_mean,
            save_figs=save_figs,
            dir_path=dir_path,
            use_total_mean=True,
            use_tkinter=False
        )
        custom_board.view()

    def close(self):
        self.simpy_env.close()

    def get_simpy_env(self):
        return self.simpy_env

    def get_last_order(self):
        return self.last_order
    
    def get_drivers(self):
        return self.simpy_env.get_drivers()
    
    def register_statistic_data(self):
        self.simpy_env.register_statistic_data()

    def get_statistics_data(self):
        return self.simpy_env.get_statistics_data()
    
    def reset_statistics(self):
        self.simpy_env.reset_statistics()
    
    def get_statistics(self):
        return self.simpy_env.compute_statistics()
    
    def get_reward_objective(self):
        return self.reward_objective
    
    def set_reward_objective(self, reward_objective: int):
        valid_objectives = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        if reward_objective not in valid_objectives:
            raise ValueError(f"Objetivo inválido! Escolha entre {valid_objectives}")
        self.reward_objective = reward_objective
    
    def get_description(self):
        descricao = []
        
        descricao.append(f"Número de motoristas: {self.num_drivers}")
        descricao.append(f"Número de estabelecimentos: {self.num_establishments}")
        descricao.append(f"Número de pedidos: {self.num_orders}")
        descricao.append(f"Número de clientes: {self.num_orders}")
        descricao.append(f"Tamanho do grid do mapa: {self.grid_map_size}")
        descricao.append(f"Objetivo da recompensa: {self.reward_objective}")
        descricao.append(f"Max Time Step: {self.max_time_step}")

        descricao.append(f"Geração de clientes e pedidos: {self.lambda_code} de {self.time_shift} em {self.time_shift} segundos")
        descricao.append(f"Porcentagem de alocação de motoristas: {self.percentage_allocation_driver}")

        descricao.append(f"Velocidade dos motorista entre: {self.vel_drivers[0]} e {self.vel_drivers[1]}")
        descricao.append(f"Tempo de preparo dos pedidos entre: {self.prepare_time[0]} e {self.prepare_time[1]} minutos")
        descricao.append(f"Raio de operação dos estabelecimentos: {self.operating_radius[0]} e {self.operating_radius[1]}")
        descricao.append(f"Capacidade de produção dos estabelecimentos: {self.production_capacity[0]} e {self.production_capacity[1]}")
        
        return "\n".join(descricao)
