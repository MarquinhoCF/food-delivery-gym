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
from food_delivery_gym.main.utils.random_manager import RandomManager
from food_delivery_gym.main.view.grid_view_pygame import GridViewPygame

class FoodDeliveryGymEnv(Env):

    REWARD_OBJECTIVES = list(range(1, 14))
    SCENARIO: dict | None = None

    @classmethod
    def set_scenario(cls, scenario_json_file_path: str) -> None:
        """
        Lê o arquivo JSON e armazena o cenário no cache de classe.
 
        Deve ser chamado no processo principal antes de criar qualquer instância do ambiente ou workers 
        de VecEnv. Em avaliações em lote com múltiplos cenários, chame novamente antes de cada grupo de
        instâncias para atualizar o cache com o novo cenário.
        """
        path = Path(scenario_json_file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Arquivo de cenário não encontrado: {path}")
        with open(path, "r", encoding="utf-8") as f:
            cls.SCENARIO = json.load(f)
    
    def set_reward_objective(self, reward_objective: int):
        if reward_objective not in self.REWARD_OBJECTIVES:
            raise ValueError(f"reward_objective deve ser um valor entre {self.REWARD_OBJECTIVES}.")
        self.reward_objective = reward_objective

    def set_mode(self, mode: EnvMode):
        self.env_mode = mode
        if self.simpy_env is not None:
            self.simpy_env.set_env_mode(mode)

    def __init__(self, scenario_json_file_path: str | None = "", reward_objective: int = 1, mode: EnvMode = EnvMode.TRAINING):
        if FoodDeliveryGymEnv.SCENARIO is None:
            if not scenario_json_file_path:
                raise ValueError(
                    "Nenhum cenário carregado. Forneça 'scenario_json_file_path' "
                    "ou chame 'FoodDeliveryGymEnv.set_scenario(path)' antes de instanciar."
                )
            FoodDeliveryGymEnv.set_scenario(scenario_json_file_path)
 
        self._load_and_validate_scenario(FoodDeliveryGymEnv.SCENARIO)
 
        self.env_mode = mode

        self.simpy_env = None # Ambiente de simulação será criado no reset
        self._last_decision_time = None # Último passo de tempo em que o agente tomou uma decisão
        self.last_simpy_env = None # Ambiente de simulação da execução anterior -> para fins de computação de estatísticas
        self.orders_generated = None # Número de pedidos que o gerador de pedidos vai gerar

        # Definindo o objetivo da recompensa
        self.set_reward_objective(reward_objective)

        # Espaço de Observação
        self.dtype_observation = np.float32
        self.observation_space = Dict({
            # --- Motoristas ---
            'drivers_coord': Box(low=0, high=self.grid_map_size - 1, shape=(self.num_drivers*2,), dtype=self.dtype_observation),
            'drivers_estimated_remaining_time': Box(low=0, high=self.max_time_step, shape=(self.num_drivers,), dtype=self.dtype_observation),
            'driver_status': Box(low=1, high=len(DriverStatus), shape=(self.num_drivers,), dtype=self.dtype_observation),
            'drivers_queue_size': Box(low=0, high=self.estimated_num_orders * 1.2, shape=(self.num_drivers,), dtype=self.dtype_observation),
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

    def _load_and_validate_scenario(self, scenario: dict):
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
        required_og = ["type", "estimated_num_orders", "time_window"]
        for k in required_og:
            if k not in og:
                raise ValueError(f"Campo obrigatório ausente em 'order_generator': '{k}'")
        if og["type"] not in ["poisson", "non_homogeneous_poisson"]:
            raise ValueError("order_generator.type deve ser 'poisson' ou 'non_homogeneous_poisson'")
        if not isinstance(og["estimated_num_orders"], int) or og["estimated_num_orders"] <= 0:
            raise ValueError("order_generator.estimated_num_orders deve ser um inteiro positivo")
        if not isinstance(og["time_window"], (int, float)) or og["time_window"] <= 0:
            raise ValueError("order_generator.time_window deve ser positivo")
        if og["type"] == "non_homogeneous_poisson":
            if "rate_function" not in og:
                raise ValueError("rate_function é obrigatório para 'non_homogeneous_poisson'")
        
        self.estimated_num_orders = og["estimated_num_orders"]
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
        required_drv = ["num", "vel", "tolerance_percentage", "max_capacity"]
        for k in required_drv:
            if k not in drv:
                raise ValueError(f"Campo obrigatório ausente em 'drivers': '{k}'")
        if not isinstance(drv["num"], int) or drv["num"] <= 0:
            raise ValueError("drivers.num deve ser um inteiro positivo")
        if not (isinstance(drv["vel"], list) and len(drv["vel"]) == 2 and all(isinstance(v, (int, float)) for v in drv["vel"])):
            raise ValueError("drivers.vel deve ser uma lista com dois números [min, max]")
        if not isinstance(drv["tolerance_percentage"], (int, float)) or drv["tolerance_percentage"] < 0:
            raise ValueError("drivers.tolerance_percentage deve ser um número não negativo")
        if not isinstance(drv["max_capacity"], int) or drv["max_capacity"] <= 0:
            raise ValueError("drivers.max_capacity deve ser um inteiro positivo")
        
        self.num_drivers = drv["num"]
        self.vel_drivers = drv["vel"]
        self.tolerance_percentage = drv["tolerance_percentage"]
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
    
    def _create_order_generator(self) -> PoissonOrderGenerator | NonHomogeneousPoissonOrderGenerator:
        generator_type = self.order_generator_config["type"]
        estimated_num_orders = self.order_generator_config["estimated_num_orders"]
        time_window = self.order_generator_config["time_window"]
        
        if generator_type == "poisson":
            return PoissonOrderGenerator(
                estimated_num_orders=estimated_num_orders,
                time_window=time_window,
                lambda_rate=self.order_generator_config.get("lambda_rate", None)
            )
        
        elif generator_type == "non_homogeneous_poisson":
            rate_function_code = self.order_generator_config["rate_function"]
            rate_function = eval(rate_function_code) # Cria a função de taxa a partir do código
            
            return NonHomogeneousPoissonOrderGenerator(
                estimated_num_orders=estimated_num_orders,
                time_window=time_window,
                rate_function=rate_function,
                max_rate=self.order_generator_config.get("max_rate", None)
            )

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
       
    def get_info(self):
        return {'simpy_time_step': self.simpy_env.now}
    
    # Avança na simulação até que um evento principal ocorra ou que a simulação termine/trunque.
    def _advance_simulation_until_event(self):
        terminated = False
        truncated = False
        core_event = None
        
        while (not terminated) and (not truncated) and (core_event is None):
            if self.simpy_env.state.get_orders_delivered() < self.orders_generated:
                self.simpy_env.step(self.render_mode)
                
                # TODO: Logs
                # # Verifica se um pedido foi entregue
                # if self.simpy_env.state.get_orders_delivered() > self.last_num_orders_delivered:
                    # print("Pedido entregue!")
                    # print(f"Número de pedidos entregues: {self.simpy_env.state.get_orders_delivered()}")
                    # self.last_num_orders_delivered = self.simpy_env.state.get_orders_delivered()

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

        # Lê as opções adicionais
        render_mode = None
        draw_grid = True
        window_size = None
        fps = 30

        if options:
            render_mode = options.get("render_mode", None)
            draw_grid = options.get("draw_grid", True)
            window_size = options.get("window_size", (1600, 1300))
            fps = options.get("fps", 30)

        self.render_mode = render_mode

        poisson_order_generator = self._create_order_generator()
        self.orders_generated = poisson_order_generator.get_number_of_orders_generated()

        # Cria o ambiente SimPy
        self.simpy_env = FoodDeliverySimpyEnv(
            map=GridMap(self.grid_map_size),
            generators=[
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
                    self.tolerance_percentage,
                    self.max_capacity,
                    self.reward_objective
                ),
                poisson_order_generator
            ],
            optimizer=None,
            view=GridViewPygame(
                grid_size=self.grid_map_size,
                draw_grid=draw_grid,
                window_size=window_size,
                fps=fps
            ) if render_mode == "human" else None
        )

        self.simpy_env.set_env_mode(self.env_mode)

        # Avança até o primeiro evento principal
        self._last_decision_time = 0
        core_event, _, _ = self._advance_simulation_until_event()
        self.current_order: Order = core_event.order if core_event else None

        observation = self.get_observation()
        info = self.get_info()

        self._last_decision_time = self.simpy_env.now

        return observation, info
    
    def reset_environment(self, seed: int | None = None, options: Optional[dict] = None):
        """
        Limpa o último estado da simulação (last_simpy_env).

        Observação:
        - Quando o ambiente está envolto com VecNormalize, o reset ocorre automaticamente ao fim de um episódio.
        - Este método garante que o último ambiente SimPy, que foi armazenado para estatísticas, seja descartado.
        """
        self.last_simpy_env = None
        return self.reset(seed=seed, options=options)
    
    def _select_driver_to_order(self, selected_driver, order):
        segment_pickup = PickupRouteSegment(order)
        segment_delivery = DeliveryRouteSegment(order)
        route = Route(self.simpy_env, [segment_pickup, segment_delivery])
        selected_driver.receive_route_requests(route)

    def _calculate_reward(self, terminated, truncated):
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

            if truncated and self.simpy_env.state.get_orders_delivered() < self.orders_generated:
                # Se a simulação foi truncada e não foram entregues todos os pedidos, penaliza a recompensa
                reward -= sum(driver.get_penality_for_late_orders() for driver in self.simpy_env.state.drivers)
        
        # Objetivo 4: Minimizar o custo de operação a partir da distância efetiva -> Recompensa negativa a cada passo
        elif self.reward_objective == 4:
            # Distância percorrida desde a última recompensa para o motorista selecionado
            reward = -sum(driver.get_and_update_distance_traveled() for driver in self.simpy_env.state.drivers)

            if truncated and self.simpy_env.state.get_orders_delivered() < self.orders_generated:
                # Se a simulação foi truncada e não foram entregues todos os pedidos, penaliza a recompensa
                reward -= (self.orders_generated - self.simpy_env.state.get_orders_delivered()) * self.simpy_env.map.max_distance() * 2

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

            if truncated and self.simpy_env.state.get_orders_delivered() < self.orders_generated:
                # Se a simulação foi truncada e não foram entregues todos os pedidos, penaliza a recompensa
                reward -= sum(driver.get_penality_for_late_orders() for driver in self.simpy_env.state.drivers)

        # Objetivo 8: Minimizar o custo de operação a partir da distância efetiva -> Recompensa negativa no fim do episódio
        elif self.reward_objective == 8 and (terminated or truncated):
            # Distância total percorrida por cada motorista
            reward = -sum(driver.get_and_update_distance_traveled() for driver in self.simpy_env.state.drivers)

            if truncated and self.simpy_env.state.get_orders_delivered() < self.orders_generated:
                # Se a simulação foi truncada e não foram entregues todos os pedidos, penaliza a recompensa
                reward -= (self.orders_generated - self.simpy_env.state.get_orders_delivered()) * self.simpy_env.map.max_distance() * 2

        # Objetivo 11: Maximizar o número de pedidos entregues -> Recompensa positiva a cada pedido entregue
        elif self.reward_objective == 11:
            reward = self.simpy_env.state.get_num_orders_delivered_since_last_check()

            # if (terminated or truncated) and (self.simpy_env.state.get_orders_delivered() == self.orders_generated):
            #     reward += 10000  # Bônus para entregar todos os pedidos

            # if (terminated or truncated) and (self.simpy_env.state.get_orders_delivered() < self.orders_generated):
            #     reward -= self.orders_generated - self.simpy_env.state.get_orders_delivered()

        # Objetivo 12: Penaliza pelo tempo total de cada pedido entregue neste step.bQuanto mais rápido o pedido for entregue, menor a penalidade (maior a recompensa).
        elif self.reward_objective == 12:
            recently_delivered = self.simpy_env.state.get_and_clear_recently_delivered_orders()
            reward = -sum(
                order.time_it_was_delivered - order.request_date
                for order in recently_delivered
            )

            if truncated and self.simpy_env.state.get_orders_delivered() < self.orders_generated:
                reward -= (self.orders_generated - self.simpy_env.state.get_orders_delivered()) * self.simpy_env.map.max_distance() * 2

        # Objetivo 13: Penaliza pelo tempo gasto de cada pedido que está na pipeline de entrega (pedidos prontos e não entregues) e de cada pedido entregue neste step. 
        elif self.reward_objective == 13:
            penalty = 0

            orders_in_delivery_pipeline = [order for order in self.simpy_env.state.orders if order.is_ready() and not order.is_delivered()]

            orders_recently_delivered = self.simpy_env.state.get_and_clear_recently_delivered_orders()

            for _ in orders_in_delivery_pipeline:
                penalty += self.simpy_env.now - self._last_decision_time
            
            for order in orders_recently_delivered:
                # Pedido foi enregue nesse intervalo, então penalidade baseada no tempo total do pedido
                penalty += order.time_it_was_delivered - self._last_decision_time

            reward = -penalty


        # if (terminated or truncated) and (self.simpy_env.state.get_orders_delivered() < self.orders_generated):
        #     # Penaliza a recompensa se o episódio terminou ou foi truncado e não foram entregues todos os pedidos
        #     reward -= (self.orders_generated - self.simpy_env.state.get_orders_delivered()) * 10000
        
        self._last_decision_time = self.simpy_env.now

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
            self._select_driver_to_order(selected_driver, self.current_order)

            core_event, terminated, truncated = self._advance_simulation_until_event()

            self.current_order = core_event.order if core_event else None

            observation = self.get_observation()

            # assert self.observation_space.contains(observation), "A observação gerada não está contida no espaço de observação."
            
            info = self.get_info()

            reward = self._calculate_reward(terminated, truncated)
            # print(f"reward: {reward}")

            if (self.env_mode != EnvMode.TRAINING) and (terminated or truncated):
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

    def close(self):
        if self.simpy_env is not None:
            self.simpy_env.close()

    def get_simpy_env(self):
        if self.last_simpy_env is not None:
            return self.last_simpy_env
        return self.simpy_env

    def get_current_order(self):
        return self.current_order
    
    def get_drivers(self):
        return self.simpy_env.get_drivers()
    
    def get_establishments(self):
        return self.simpy_env.get_establishments()
    
    def get_num_orders_generated(self):
        return self.orders_generated
    
    def get_reward_objective(self):
        return self.reward_objective
    
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

        descricao.append("=== Configuração do Ambiente de Entrega ===")

        # Dimensões fundamentais
        descricao.append(f"- Número de motoristas: {self.num_drivers}")
        descricao.append(f"- Número de estabelecimentos: {self.num_establishments}")
        descricao.append(f"- Tamanho do grid do mapa: {self.grid_map_size}x{self.grid_map_size}")

        # Parâmetros operacionais
        descricao.append(f"- Objetivo da função de recompensa: {self.reward_objective}")
        descricao.append(f"- Tempo máximo de simulação (max_time_step): {self.max_time_step} minutos")

        # Parâmetros de geração de pedidos
        if self.order_generator_config["type"] == "poisson":
            descricao.append("- Geração de pedidos: Processo de Poisson Homogêneo")
            descricao.append(f"  • {self.order_generator_config['estimated_num_orders']} pedidos estimados em {self.order_generator_config['time_window']} minutos")
            if self.order_generator_config.get('lambda_rate', None) is not None:
                descricao.append(f"  • Taxa λ: {self.order_generator_config['lambda_rate']} pedidos por minuto")

        elif self.order_generator_config["type"] == "non_homogeneous_poisson":
            descricao.append("- Geração de pedidos: Poisson Não Homogêneo")
            descricao.append(f"  • {self.order_generator_config['estimated_num_orders']} pedidos por {self.order_generator_config['time_window']} minutos")
            descricao.append(f"  • Função de taxa: {self.order_generator_config['rate_function']}")
            if self.order_generator_config.get("max_rate", None) is not None:
                descricao.append(f"  • Taxa máxima: {self.order_generator_config['max_rate']} pedidos por minuto")
            
        # Parâmetros dos motoristas
        descricao.append("- Motoristas:")
        descricao.append(f"  • Velocidade dos motoristas: entre {self.vel_drivers[0]} e {self.vel_drivers[1]} unidades/min")
        descricao.append(f"  • Tolerância de piora de tempo de entrega (%): {self.tolerance_percentage}%")
        descricao.append(f"  • Capacidade máxima: {self.max_capacity}")

        # Parâmetros dos estabelecimentos
        descricao.append("- Estabelecimentos:")
        descricao.append(f"  • Raio de operação: entre {self.operating_radius[0]} e {self.operating_radius[1]} unidades")
        descricao.append(f"  • Tempo de preparo dos pedidos: entre {self.prepare_time[0]} e "f"{self.prepare_time[1]} minutos")
        descricao.append(f"  • Capacidade de produção: entre {self.production_capacity[0]} e "f"{self.production_capacity[1]} pedidos simultâneos")
        descricao.append(f"  • Porcentagem de conclusão do pedido para alocação do motorista: {self.percentage_allocation_driver}%")

        return "\n".join(descricao)