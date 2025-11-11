import json
from pathlib import Path
import traceback
from typing import Optional
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Dict, Box, Discrete

from food_delivery_gym.main.driver.driver_status import DriverStatus
from food_delivery_gym.main.environment.env_mode import EnvMode
from food_delivery_gym.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv
from food_delivery_gym.main.generator.initial_dynamic_route_driver_generator import InitialDynamicRouteDriverGenerator
from food_delivery_gym.main.generator.initial_establishment_order_rate_generator import InitialEstablishmentOrderRateGenerator
from food_delivery_gym.main.generator.poisson_order_generator import PoissonOrderGenerator
from food_delivery_gym.main.generator.non_homogeneous_poisson_order_generator import NonHomogeneousPoissonOrderGenerator
from food_delivery_gym.main.map.grid_map import GridMap
from food_delivery_gym.main.order.order import Order
from food_delivery_gym.main.route.delivery_route_segment import DeliveryRouteSegment
from food_delivery_gym.main.route.pickup_route_segment import PickupRouteSegment
from food_delivery_gym.main.route.route import Route
from food_delivery_gym.main.statistic.driver_idle_time_metric import DriverIdleTimeMetric
from food_delivery_gym.main.statistic.driver_time_waiting_for_order_metric import DriverTimeWaitingForOrderMetric
from food_delivery_gym.main.statistic.poisson_order_generation_metric import PoissonOrderGenerationMetric
from food_delivery_gym.main.statistic.summarized_data_board import SummarizedDataBoard
from food_delivery_gym.main.statistic.driver_orders_delivered_metric import DriverOrdersDeliveredMetric
from food_delivery_gym.main.statistic.driver_total_distance_metric import DriverTotalDistanceMetric
from food_delivery_gym.main.statistic.establishment_active_time_metric import EstablishmentActiveTimeMetric
from food_delivery_gym.main.statistic.driver_time_spent_on_delivery import DriverTimeSpentOnDelivery
from food_delivery_gym.main.statistic.establishment_max_orders_in_queue_metric import EstablishmentMaxOrdersInQueueMetric
from food_delivery_gym.main.statistic.establishment_orders_fulfilled_metric import EstablishmentOrdersFulfilledMetric
from food_delivery_gym.main.statistic.order_curve_metric import OrderCurveMetric
from food_delivery_gym.main.utils.random_manager import RandomManager
from food_delivery_gym.main.view.grid_view_pygame import GridViewPygame

class FoodDeliveryGymEnv(Env):

    def __init__(self, scenario_json_file_path = str, reward_objective: int = 1):

        path = Path(scenario_json_file_path)

        if not path.is_file():
            raise FileNotFoundError(f"Scenario file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            scenario = json.load(f)

        self.read_scenario_json(scenario)

        self.env_mode = EnvMode.TRAINING

        self.simpy_env = None # Ambiente de simulação será criado no reset
        self.last_simpy_env = None # Ambiente de simulação da execução anterior -> para fins de computação de estatísticas

        # Definindo o objetivo da recompensa
        if reward_objective not in range(1, 11):
            raise ValueError("reward_objective deve ser um valor entre 1 e 10.")
        self.set_reward_objective(reward_objective)

        # Espaço de Observação
        self.dtype_observation = np.int32
        self.observation_space = Dict({
            # --- Motoristas ---
            'drivers_coord': Box(low=0, high=self.grid_map_size - 1, shape=(self.num_drivers*2,), dtype=self.dtype_observation),
            'drivers_estimated_remaining_time': Box(low=0, high=self.max_time_step, shape=(self.num_drivers,), dtype=self.dtype_observation),
            'driver_status': Box(low=1, high=len(DriverStatus), shape=(self.num_drivers,), dtype=self.dtype_observation),
            'drivers_queue_size': Box(low=0, high=self.num_orders, shape=(self.num_drivers,), dtype=self.dtype_observation),
            'drivers_velocity': Box(low=min(self.vel_drivers), high=max(self.vel_drivers), shape=(self.num_drivers,), dtype=self.dtype_observation),

            # --- Pedido Atual ---  
            'order_restaurant_coord': Box(low=0, high=self.grid_map_size - 1, shape=(2,), dtype=self.dtype_observation),
            'order_customer_coord': Box(low=0, high=self.grid_map_size - 1, shape=(2,), dtype=self.dtype_observation),
            'order_estimated_ready_time': Box(low=0, high=self.max_time_step, shape=(1,), dtype=self.dtype_observation),
            'order_estimated_delivery_time': Box(low=0, high=self.max_time_step, shape=(self.num_drivers,), dtype=self.dtype_observation),

            # --- Ambiente ---
            'current_time_step': Box(low=0, high=self.max_time_step, shape=(1,), dtype=self.dtype_observation)
        })

        # Espaço de Ação
        self.action_space = Discrete(self.num_drivers)  # Escolher qual driver pegará o pedido

    def read_scenario_json(self, scenario: dict):
        # Estrutura esperada
        required_sections = ["order_generator", "simpy_env", "grid_map", "drivers", "establishments"]
        for section in required_sections:
            if section not in scenario:
                raise ValueError(f"Seção obrigatória ausente: '{section}'")

        og = scenario["order_generator"]
        env = scenario["simpy_env"]
        grid = scenario["grid_map"]
        drv = scenario["drivers"]
        est = scenario["establishments"]

        # 1. Order Generator
        required_og = ["type", "total_orders", "time_window"]
        for k in required_og:
            if k not in og:
                raise ValueError(f"Campo obrigatório ausente em 'order_generator': '{k}'")
        if og["type"] not in ["poisson", "non_homogeneous_poisson"]:
            raise ValueError("order_generator.type deve ser 'poisson' ou 'non_homogeneous_poisson'")
        if not isinstance(og["total_orders"], int) or og["total_orders"] <= 0:
            raise ValueError("order_generator.total_orders deve ser um inteiro positivo")
        if not isinstance(og["time_window"], (int, float)) or og["time_window"] <= 0:
            raise ValueError("order_generator.time_window deve ser positivo")
        if og["type"] == "non_homogeneous_poisson":
            if "rate_function" not in og:
                raise ValueError("rate_function é obrigatório para 'non_homogeneous_poisson'")
        
        self.num_orders = og["total_orders"]
        self.order_generator_config = scenario.get("order_generator", {})

        # 2. simpy_env
        if "max_time_step" not in env:
            raise ValueError("Campo obrigatório ausente em 'simpy_env': 'max_time_step'")
        if not isinstance(env["max_time_step"], (int, float)) or env["max_time_step"] <= 0:
            raise ValueError("simpy_env.max_time_step deve ser um número positivo")
        
        self.max_time_step = env["max_time_step"]

        # 3. grid_map
        if "size" not in grid:
            raise ValueError("Campo obrigatório ausente em 'grid_map': 'size'")
        if not isinstance(grid["size"], int) or grid["size"] <= 0:
            raise ValueError("grid_map.size deve ser um inteiro positivo")
        
        self.grid_map_size = grid["size"]

        # 4. drivers
        required_drv = ["num", "vel", "max_delay_percentage", "max_capacity"]
        for k in required_drv:
            if k not in drv:
                raise ValueError(f"Campo obrigatório ausente em 'drivers': '{k}'")
        if not isinstance(drv["num"], int) or drv["num"] <= 0:
            raise ValueError("drivers.num deve ser um inteiro positivo")
        if not (isinstance(drv["vel"], list) and len(drv["vel"]) == 2 and all(isinstance(v, (int, float)) for v in drv["vel"])):
            raise ValueError("drivers.vel deve ser uma lista com dois números [min, max]")
        if not isinstance(drv["max_delay_percentage"], (int, float)) or drv["max_delay_percentage"] < 0:
            raise ValueError("drivers.max_delay_percentage deve ser um número não negativo")
        if not isinstance(drv["max_capacity"], int) or drv["max_capacity"] <= 0:
            raise ValueError("drivers.max_capacity deve ser um inteiro positivo")
        
        self.num_drivers = drv["num"]
        self.vel_drivers = drv["vel"]
        self.max_delay_percentage = drv["max_delay_percentage"]
        self.max_capacity = drv["max_capacity"]

        # 5. establishments
        required_est = ["num", "prepare_time", "operating_radius", "production_capacity", "percentage_allocation_driver"]
        for k in required_est:
            if k not in est:
                raise ValueError(f"Campo obrigatório ausente em 'establishments': '{k}'")
        if not isinstance(est["num"], int) or est["num"] <= 0:
            raise ValueError("establishments.num deve ser um inteiro positivo")
        for key in ["prepare_time", "operating_radius", "production_capacity"]:
            value = est[key]
            if not (isinstance(value, list) and len(value) == 2 and all(isinstance(v, (int, float)) for v in value)):
                raise ValueError(f"'{key}' deve ser uma lista com dois valores numéricos [min, max]")
            if value[0] > value[1]:
                raise ValueError(f"'{key}' deve estar em ordem crescente (min <= max)")
        pad = est["percentage_allocation_driver"]
        if not (isinstance(pad, (int, float)) and 0 <= pad <= 1):
            raise ValueError("establishments.percentage_allocation_driver deve ser um número entre 0 e 1")
        
        self.num_establishments = est["num"]
        self.prepare_time = est["prepare_time"]
        self.operating_radius = est["operating_radius"]
        self.production_capacity = est["production_capacity"]
        self.percentage_allocation_driver = est["percentage_allocation_driver"]
    
    def _create_order_generator(self):
        if not self.order_generator_config:
            return None
            
        generator_type = self.order_generator_config.get("type", "poisson")
        total_orders = self.num_orders
        time_window = self.order_generator_config.get("time_window", self.max_time_step)
        
        if generator_type == "poisson":
            return PoissonOrderGenerator(
                total_orders=total_orders,
                time_window=time_window,
                lambda_rate=self.order_generator_config.get("lambda_rate", None)
            )
        
        elif generator_type == "non_homogeneous_poisson":
            rate_function_code = self.order_generator_config.get("rate_function")
            rate_function = eval(rate_function_code) # Cria a função de taxa a partir do código
            
            return NonHomogeneousPoissonOrderGenerator(
                total_orders=total_orders,
                time_window=time_window,
                rate_function=rate_function,
                max_rate=self.order_generator_config.get("max_rate", None)
            )
        
        else:
            raise ValueError(f"Tipo de gerador não suportado: {generator_type}")

    def get_observation(self):
        # --- Motoristas ---
        # 1. drivers_coord: Coordenada atual de cada motorista (posição no grid)
        drivers_coord = np.zeros((self.num_drivers*2,), dtype=self.dtype_observation)
        for i, driver in enumerate(self.simpy_env.state.drivers):
            coords = driver.get_coordinate()
            index = i * 2
            drivers_coord[index] = coords[0]
            drivers_coord[index + 1] = coords[1]
        # 2. drivers_estimated_remaining_time: Tempo estimado restante para cada motorista completar todas as entregas na sua lista
        drivers_estimated_remaining_time = np.zeros((self.num_drivers,), dtype=self.dtype_observation)
        for i, driver in enumerate(self.simpy_env.state.drivers):
            drivers_estimated_remaining_time[i] = driver.estimate_total_busy_time()
        # 3. driver_status: Status atual de cada motorista (disponível, coletando, entregando, etc.)
        driver_status = np.zeros((self.num_drivers,), dtype=self.dtype_observation)
        for i, driver in enumerate(self.simpy_env.state.drivers):
            driver_status[i] = driver.get_status_for_observation().value
        # 4. drivers_queue_size: Número de pedidos na lista de cada motorista
        drivers_queue_size = np.zeros((self.num_drivers,), dtype=self.dtype_observation)
        for i, driver in enumerate(self.simpy_env.state.drivers):
            drivers_queue_size[i] = driver.get_number_of_orders_in_list()
        # 5. drivers_velocity: Velocidade de cada motorista
        drivers_velocity = np.zeros((self.num_drivers,), dtype=self.dtype_observation)
        for i, driver in enumerate(self.simpy_env.state.drivers):
            drivers_velocity[i] = driver.get_velocity()

        # --- Pedido Atual ---
        # 1. order_restaurant_coord: Coordenada do restaurante do pedido atual
        order_restaurant_coord = np.zeros((2,), dtype=self.dtype_observation)
        if self.current_order:
            order_restaurant_coord[0] = self.current_order.get_establishment().get_coordinate()[0]
            order_restaurant_coord[1] = self.current_order.get_establishment().get_coordinate()[1]
        # 2. order_customer_coord: Coordenada do cliente do pedido atual
        order_customer_coord = np.zeros((2,), dtype=self.dtype_observation)
        if self.current_order:
            order_customer_coord[0] = self.current_order.get_customer().get_coordinate()[0]
            order_customer_coord[1] = self.current_order.get_customer().get_coordinate()[1]
        # 3. order_estimated_ready_time: Tempo estimado para o pedido atual ficar pronto no restaurante
        order_estimated_ready_time = np.zeros((1,), dtype=self.dtype_observation)
        if self.current_order:
            order_estimated_ready_time[0] = self.current_order.get_estimated_ready_time()
        # 4. order_estimated_delivery_time: Tempo estimado para o pedido atual ser entregue ao cliente
        order_estimated_delivery_time = np.zeros((self.num_drivers,), dtype=self.dtype_observation)
        for i, driver in enumerate(self.simpy_env.state.drivers):
                    order_estimated_delivery_time[i] = driver.estimate_time_to_complete_next_order(self.current_order)

        # --- Ambiente ---
        # 1. current_time_step: O tempo atual da simulação (número do passo)
        current_time_step = self.simpy_env.now
        current_time_step = np.array([current_time_step], dtype=self.dtype_observation)

        # Criando a observação final no formato esperado
        obs = {
            'drivers_coord': drivers_coord,
            'drivers_estimated_remaining_time': drivers_estimated_remaining_time,
            'driver_status': driver_status,
            'drivers_queue_size': drivers_queue_size,
            'drivers_velocity': drivers_velocity,
            'order_restaurant_coord': order_restaurant_coord,
            'order_customer_coord': order_customer_coord,
            'order_estimated_ready_time': order_estimated_ready_time,
            'order_estimated_delivery_time': order_estimated_delivery_time,
            'current_time_step': current_time_step
        }

        return obs
       
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

        generators = [
            InitialEstablishmentOrderRateGenerator(
                self.num_establishments, 
                self.prepare_time, 
                self.operating_radius, 
                self.production_capacity,
                self.percentage_allocation_driver, 
            ),
            InitialDynamicRouteDriverGenerator(
                self.num_drivers,
                self.vel_drivers,
                self.max_delay_percentage,
                self.max_capacity,
                self.reward_objective
            )
        ]

        # Adiciona o gerador de pedidos configurado
        order_generator = self._create_order_generator()
        if order_generator is not None:
            generators.append(order_generator)

        self.simpy_env = FoodDeliverySimpyEnv(
            map=GridMap(self.grid_map_size),
            generators=generators,
            optimizer=None,
            view=GridViewPygame(grid_size=self.grid_map_size) if self.render_mode == "human" else None
        )

        # self.last_num_orders_delivered = 0
        core_event, _, _ = self.advance_simulation_until_event()
        self.current_order: Order = core_event.order if core_event else None

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

        # Objetivo 1: Minimizar o tempo de entrega a partir da expectativa de tempo gasto com a entrega -> Recompensa negativa a cada passo
        if self.reward_objective == 1:
            # Soma das estimativas do tempo de ocupação de cada motoristas
            reward = -sum(driver.estimate_total_busy_time() for driver in self.simpy_env.state.drivers)

        # Objetivo 2: Minimizar o custo de operação a partir da expectativa da distância a ser percorrida -> Recompensa negativa a cada passo
        if self.reward_objective == 2:
            # Soma das estimativas do tempo de ocupação de cada motoristas
            reward = -sum(driver.calculate_total_distance_to_travel() for driver in self.simpy_env.state.drivers)

        # Objetivo 3: Minimizar o tempo de entrega dos motoristas a partir do tempo efetivo gasto -> Recompensa negativa a cada passo
        # Objetivo 9: Minimizar o tempo de entrega dos motoristas a partir do tempo efetivo gasto (Com penalização 5x para pedidos não coletados) -> Recompensa negativa a cada passo
        elif self.reward_objective in [3, 9]:
            # Soma do tempo efetivo gasto por cada motorista
            reward = -sum(driver.get_penality_for_time_spent_for_delivery() for driver in self.simpy_env.state.drivers)

            if truncated and self.simpy_env.state.orders_delivered < self.num_orders:
                # Se a simulação foi truncada e não foram entregues todos os pedidos, penaliza a recompensa
                reward -= sum(driver.get_penality_for_late_orders() for driver in self.simpy_env.state.drivers)
        
        # Objetivo 4: Minimizar o custo de operação a partir da distância efetiva -> Recompensa negativa a cada passo
        elif self.reward_objective == 4:
            # Distância percorrida desde a última recompensa para o motorista selecionado
            reward = -sum(driver.get_and_update_distance_traveled() for driver in self.simpy_env.state.drivers)

            if truncated and self.simpy_env.state.orders_delivered < self.num_orders:
                # Se a simulação foi truncada e não foram entregues todos os pedidos, penaliza a recompensa
                reward -= (self.num_orders - self.simpy_env.state.orders_delivered) * self.simpy_env.map.max_distance() * 2

        # Objetivo 5: Minimizar o tempo de entrega a partir da expectativa de tempo gasto com a entrega -> Recompensa negativa ao final do episódio
        elif self.reward_objective == 5:
            # Atualiza as estimativas de tempo de ocupação de cada motorista
            for driver in self.simpy_env.state.drivers:
                driver.update_expected_delivery_time_reward()

            if terminated or truncated:
                # Penalidade baseada na soma das estimativas de tempo de ocupação dos motoristas
                reward = -sum(driver.get_expected_delivery_time_reward() for driver in self.simpy_env.state.drivers)

        # Objetivo 6: Minimizar o custo de operação a partir da expectativa da distância a ser percorrida -> Recompensa negativa ao final do episódio
        elif self.reward_objective == 6:
            # Atualiza as estimativas de tempo de ocupação de cada motorista
            for driver in self.simpy_env.state.drivers:
                driver.update_distance_to_be_traveled_reward()

            if terminated or truncated:
                # Penalidade baseada na soma das estimativas de tempo de ocupação dos motoristas
                reward = -sum(driver.get_distance_to_be_traveled_reward() for driver in self.simpy_env.state.drivers)

        # Objetivo 7: Minimizar o tempo de entrega dos motoristas a partir do tempo efetivo gasto -> Recompensa negativa no fim do episódio
        # Objetivo 10: Minimizar o tempo de entrega dos motoristas a partir do tempo efetivo gasto (Com penalização 5x para pedidos não coletados) -> Recompensa negativa no fim do episódio
        elif self.reward_objective in [7, 10] and (terminated or truncated):
            # Soma do tempo efetivo gasto por cada motorista
            reward = -sum(driver.get_penality_for_time_spent_for_delivery() for driver in self.simpy_env.state.drivers)

            if truncated and self.simpy_env.state.orders_delivered < self.num_orders:
                # Se a simulação foi truncada e não foram entregues todos os pedidos, penaliza a recompensa
                reward -= sum(driver.get_penality_for_late_orders() for driver in self.simpy_env.state.drivers)

        # Objetivo 8: Minimizar o custo de operação a partir da distância efetiva -> Recompensa negativa no fim do episódio
        elif self.reward_objective == 8 and (terminated or truncated):
            # Distância total percorrida por cada motorista
            reward = -sum(driver.get_and_update_distance_traveled() for driver in self.simpy_env.state.drivers)

            if truncated and self.simpy_env.state.orders_delivered < self.num_orders:
                # Se a simulação foi truncada e não foram entregues todos os pedidos, penaliza a recompensa
                reward -= (self.num_orders - self.simpy_env.state.orders_delivered) * self.simpy_env.map.max_distance() * 2

        if (terminated or truncated) and (self.simpy_env.state.orders_delivered < self.num_orders):
            # Penaliza a recompensa se o episódio terminou ou foi truncado e não foram entregues todos os pedidos
            reward -= (self.num_orders - self.simpy_env.state.orders_delivered) * 10000
        
        return reward
        
    def step(self, action):
        try:
            if action < 0 or action >= self.num_drivers:
                raise ValueError(f"A ação {action} é inválida! Deve ser um número entre 0 e {self.num_drivers}")

            if self.render_mode == "human":
                truncated = self.simpy_env.view.quited

            terminated = False
            # print("action: {}".format(action))
            # print("current_order: {}".format(vars(self.current_order)))
            selected_driver = self.simpy_env.state.drivers[action]
            self.select_driver_to_order(selected_driver, self.current_order)

            core_event, terminated, truncated = self.advance_simulation_until_event()

            self.current_order = core_event.order if core_event else None

            observation = self.get_observation()

            # assert self.observation_space.contains(observation), "A observação gerada não está contida no espaço de observação."
            
            info = self._get_info()

            reward = self.calculate_reward(terminated, truncated)
            # print(f"reward: {reward}")

            if (self.env_mode != EnvMode.TRAINING) and (terminated or truncated):
                self.register_statistic_data()
                self.last_simpy_env = self.simpy_env

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

    def show_statistics_board(self, sum_reward = None, dir_path = None):
        if self.env_mode == EnvMode.EVALUATING:
            if self.last_simpy_env == None:
                raise ValueError(
                    "Dados de simulação indisponíveis. Certifique-se de que o ambiente foi executado ao menos uma vez "
                    "e que o método 'reset_last_simpy_env' não foi chamado antes da coleta ou exibição das estatísticas."
                )
            simpy_env = self.last_simpy_env
        else:
            simpy_env = self.simpy_env

        if sum_reward is None and dir_path is None:
            save_figs = False
        else:
            save_figs = True
        
        custom_board = SummarizedDataBoard(metrics=[
            PoissonOrderGenerationMetric(simpy_env),
            EstablishmentOrdersFulfilledMetric(simpy_env),
            EstablishmentMaxOrdersInQueueMetric(simpy_env),
            EstablishmentActiveTimeMetric(simpy_env),
            DriverTimeSpentOnDelivery(simpy_env),
            DriverOrdersDeliveredMetric(simpy_env),
            DriverTotalDistanceMetric(simpy_env),
            DriverIdleTimeMetric(simpy_env),
            DriverTimeWaitingForOrderMetric(simpy_env)
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
            DriverTimeSpentOnDelivery(self.simpy_env, drivers_statistics=statistics["drivers"]),
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
        if self.simpy_env is not None:
            self.simpy_env.close()

    def get_simpy_env(self):
        return self.simpy_env

    def get_current_order(self):
        return self.current_order
    
    def get_drivers(self):
        return self.simpy_env.get_drivers()
    
    def register_statistic_data(self):
        self.simpy_env.register_statistic_data()

    def get_statistics_data(self):
        return self.simpy_env.get_statistics_data()
    
    def reset_last_simpy_env(self):
        """
        Limpa o último estado da simulação (last_simpy_env).

        Observação:
        - Quando o ambiente está envolto com VecNormalize, o reset ocorre automaticamente ao fim de um episódio.
        - Este método garante que o último ambiente SimPy, que foi armazenado para estatísticas, seja descartado.
        """
        self.last_simpy_env = None
    
    def reset_statistics(self):
        self.simpy_env.reset_statistics()
        self.reset_last_simpy_env()
    
    def get_statistics(self):
        return self.simpy_env.compute_statistics()
    
    def get_reward_objective(self):
        return self.reward_objective
    
    def set_reward_objective(self, reward_objective: int):
        valid_objectives = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        if reward_objective not in valid_objectives:
            raise ValueError(f"Objetivo inválido! Escolha entre {valid_objectives}")
        self.reward_objective = reward_objective
    
    def print_enviroment_state(self, options = None):
        if options is None:
            options = {
                "customers": False,
                "establishments": True,
                "drivers": True,
                "orders": False,
                "events": False,
                "orders_delivered": True
            }
        if self.current_order:
            print(f'current_order:\n{self.current_order.__str__()}')
        self.simpy_env.print_enviroment_state(options=options)
    
    def get_description(self):
        descricao = []
        
        descricao.append(f"Número de motoristas: {self.num_drivers}")
        descricao.append(f"Número de estabelecimentos: {self.num_establishments}")
        descricao.append(f"Número de pedidos: {self.num_orders}")
        descricao.append(f"Número de clientes: {self.num_orders}")
        descricao.append(f"Tamanho do grid do mapa: {self.grid_map_size}")
        descricao.append(f"Objetivo da recompensa: {self.reward_objective}")
        descricao.append(f"Max Time Step: {self.max_time_step}")

        descricao.append(f"Geração de clientes e pedidos: {self.lambda_code} de {self.time_shift} em {self.time_shift} minutos")
        descricao.append(f"Porcentagem de alocação de motoristas: {self.percentage_allocation_driver}")

        descricao.append(f"Velocidade dos motorista entre: {self.vel_drivers[0]} e {self.vel_drivers[1]}")
        descricao.append(f"Tempo de preparo dos pedidos entre: {self.prepare_time[0]} e {self.prepare_time[1]} minutos")
        descricao.append(f"Raio de operação dos estabelecimentos: {self.operating_radius[0]} e {self.operating_radius[1]}")
        descricao.append(f"Capacidade de produção dos estabelecimentos: {self.production_capacity[0]} e {self.production_capacity[1]}")
        
        return "\n".join(descricao)
