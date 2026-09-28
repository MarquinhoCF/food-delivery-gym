from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from food_delivery_gym.main.utils.rate_function_utils import build_rate_function, validate_rate_function

DRIVER_TYPES = ("dynamic_route", "driver", "capacity")
ESTABLISHMENT_TYPES = ("order_rate", "establishment")
ORDER_GENERATOR_TYPES = ("poisson", "non_homogeneous_poisson")


@dataclass(frozen=True)
class ScenarioSpec:
    # Order generator
    order_generator_config: dict[str, Any]
    estimated_num_orders: int

    # Env / map
    max_time_step: float
    grid_map_size: int

    # Drivers
    driver_type: str
    num_drivers: int
    vel_drivers: list
    tolerance_percentage: float | None
    max_capacity: int | None
    bag_capacity: list | None  # [length, height, width, weight] para capacity drivers

    # Establishments
    establishment_type: str
    num_establishments: int
    prepare_time: list | None
    operating_radius: list | None
    production_capacity: list | None
    percentage_allocation_driver: float | None
    use_estimate: bool

    def build_order_generator(self, rng=None):
        from food_delivery_gym.main.generator.poisson_order_generator import PoissonOrderGenerator
        from food_delivery_gym.main.generator.non_homogeneous_poisson_order_generator import (
            NonHomogeneousPoissonOrderGenerator,
        )

        cfg = self.order_generator_config
        generator_type = cfg["type"]
        estimated_num_orders = cfg["estimated_num_orders"]
        time_window = cfg["time_window"]

        if generator_type == "poisson":
            return PoissonOrderGenerator(
                estimated_num_orders=estimated_num_orders,
                time_window=time_window,
                lambda_rate=cfg.get("lambda_rate", None),
                rng=rng,
            )

        if generator_type == "non_homogeneous_poisson":
            rate_function = build_rate_function(cfg["rate_function"])
            return NonHomogeneousPoissonOrderGenerator(
                estimated_num_orders=estimated_num_orders,
                time_window=time_window,
                rate_function=rate_function,
                max_rate=cfg.get("max_rate", None),
                rng=rng,
            )

        raise ValueError(f"order_generator.type desconhecido: '{generator_type}'")

    def build_establishment_generator(self, rng=None):
        if self.establishment_type == "order_rate":
            from food_delivery_gym.main.generator.initial_establishment_order_rate_generator import (
                InitialEstablishmentOrderRateGenerator,
            )
            return InitialEstablishmentOrderRateGenerator(
                self.num_establishments,
                self.prepare_time,
                self.operating_radius,
                self.production_capacity,
                self.percentage_allocation_driver,
                rng=rng,
            )

        if self.establishment_type == "establishment":
            from food_delivery_gym.main.generator.initial_establishment_generator import (
                InitialEstablishmentGenerator,
            )
            return InitialEstablishmentGenerator(
                self.num_establishments,
                use_estimate=self.use_estimate,
                rng=rng,
            )

        raise ValueError(f"establishments.type desconhecido: '{self.establishment_type}'")

    def build_driver_generator(self, reward_objective: int, rng=None):
        if self.driver_type == "dynamic_route":
            from food_delivery_gym.main.generator.initial_dynamic_route_driver_generator import (
                InitialDynamicRouteDriverGenerator,
            )
            return InitialDynamicRouteDriverGenerator(
                self.num_drivers,
                self.vel_drivers,
                self.tolerance_percentage,
                self.max_capacity,
                reward_objective,
                rng=rng,
            )

        if self.driver_type == "driver":
            from food_delivery_gym.main.generator.initial_driver_generator import (
                InitialDriverGenerator,
            )
            return InitialDriverGenerator(
                self.num_drivers,
                self.vel_drivers,
                reward_objective,
                rng=rng,
            )

        if self.driver_type == "capacity":
            from food_delivery_gym.main.generator.initial_capacity_driver_generator import (
                InitialCapacityDriverGenerator,
            )
            return InitialCapacityDriverGenerator(
                self.num_drivers,
                self.vel_drivers,
                self.bag_capacity,
                reward_objective,
                rng=rng,
            )

        raise ValueError(f"drivers.type desconhecido: '{self.driver_type}'")


def parse_scenario(scenario: dict) -> ScenarioSpec:
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
    if og["type"] not in ORDER_GENERATOR_TYPES:
        raise ValueError(
            "order_generator.type deve ser 'poisson' ou 'non_homogeneous_poisson'"
        )
    if not isinstance(og["estimated_num_orders"], int) or og["estimated_num_orders"] <= 0:
        raise ValueError("order_generator.estimated_num_orders deve ser um inteiro positivo")
    if not isinstance(og["time_window"], (int, float)) or og["time_window"] <= 0:
        raise ValueError("order_generator.time_window deve ser positivo")
    if og["type"] == "non_homogeneous_poisson":
        if "rate_function" not in og:
            raise ValueError("rate_function é obrigatório para 'non_homogeneous_poisson'")
        validate_rate_function(og["rate_function"])

    # 2. simpy_env
    if "max_time_step" not in env:
        raise ValueError("Campo obrigatório ausente em 'simpy_env': 'max_time_step'")
    if not isinstance(env["max_time_step"], (int, float)) or env["max_time_step"] <= 0:
        raise ValueError("simpy_env.max_time_step deve ser um número positivo")

    # 3. grid_map
    if "size" not in grid:
        raise ValueError("Campo obrigatório ausente em 'grid_map': 'size'")
    if not isinstance(grid["size"], int) or grid["size"] <= 0:
        raise ValueError("grid_map.size deve ser um inteiro positivo")

    # 4. drivers
    if "type" not in drv:
        raise ValueError("Campo obrigatório ausente em 'drivers': 'type'")
    driver_type = drv["type"]
    if driver_type not in DRIVER_TYPES:
        raise ValueError(
            f"drivers.type inválido: '{driver_type}'. "
            f"Opções: {list(DRIVER_TYPES)}"
        )

    required_drv_common = ["num", "vel"]
    for k in required_drv_common:
        if k not in drv:
            raise ValueError(f"Campo obrigatório ausente em 'drivers': '{k}'")
    if not isinstance(drv["num"], int) or drv["num"] <= 0:
        raise ValueError("drivers.num deve ser um inteiro positivo")
    if not (
        isinstance(drv["vel"], list)
        and len(drv["vel"]) == 2
        and all(isinstance(v, (int, float)) for v in drv["vel"])
    ):
        raise ValueError("drivers.vel deve ser uma lista com dois números [min, max]")

    tolerance_percentage: float | None = None
    max_capacity: int | None = None
    bag_capacity: list | None = None

    if driver_type == "dynamic_route":
        for k in ("tolerance_percentage", "max_capacity"):
            if k not in drv:
                raise ValueError(f"Campo obrigatório ausente em 'drivers': '{k}'")
        if not isinstance(drv["tolerance_percentage"], (int, float)) or drv["tolerance_percentage"] < 0:
            raise ValueError("drivers.tolerance_percentage deve ser um número não negativo")
        if not isinstance(drv["max_capacity"], int) or drv["max_capacity"] <= 0:
            raise ValueError("drivers.max_capacity deve ser um inteiro positivo")
        tolerance_percentage = drv["tolerance_percentage"]
        max_capacity = drv["max_capacity"]

    elif driver_type == "capacity":
        if "capacity" not in drv:
            raise ValueError("Campo obrigatório ausente em 'drivers': 'capacity'")
        cap = drv["capacity"]
        if not (
            isinstance(cap, list)
            and len(cap) == 4
            and all(isinstance(v, (int, float)) for v in cap)
        ):
            raise ValueError(
                "drivers.capacity deve ser uma lista [length, height, width, weight]"
            )
        bag_capacity = list(cap)

    # 5. establishments
    if "type" not in est:
        raise ValueError("Campo obrigatório ausente em 'establishments': 'type'")
    establishment_type = est["type"]
    if establishment_type not in ESTABLISHMENT_TYPES:
        raise ValueError(
            f"establishments.type inválido: '{establishment_type}'. "
            f"Opções: {list(ESTABLISHMENT_TYPES)}"
        )

    if "num" not in est:
        raise ValueError("Campo obrigatório ausente em 'establishments': 'num'")
    if not isinstance(est["num"], int) or est["num"] <= 0:
        raise ValueError("establishments.num deve ser um inteiro positivo")

    prepare_time: list | None = None
    operating_radius: list | None = None
    production_capacity: list | None = None
    percentage_allocation_driver: float | None = None
    use_estimate = bool(est.get("use_estimate", True))

    if establishment_type == "order_rate":
        required_est = [
            "prepare_time",
            "operating_radius",
            "production_capacity",
            "percentage_allocation_driver",
        ]
        for k in required_est:
            if k not in est:
                raise ValueError(f"Campo obrigatório ausente em 'establishments': '{k}'")
        for key in ["prepare_time", "operating_radius", "production_capacity"]:
            value = est[key]
            if not (
                isinstance(value, list)
                and len(value) == 2
                and all(isinstance(v, (int, float)) for v in value)
            ):
                raise ValueError(f"'{key}' deve ser uma lista com dois valores numéricos [min, max]")
            if value[0] > value[1]:
                raise ValueError(f"'{key}' deve estar em ordem crescente (min <= max)")
        pad = est["percentage_allocation_driver"]
        if not (isinstance(pad, (int, float)) and 0 <= pad <= 1):
            raise ValueError(
                "establishments.percentage_allocation_driver deve ser um número entre 0 e 1"
            )
        prepare_time = est["prepare_time"]
        operating_radius = est["operating_radius"]
        production_capacity = est["production_capacity"]
        percentage_allocation_driver = pad

    return ScenarioSpec(
        order_generator_config=dict(og),
        estimated_num_orders=og["estimated_num_orders"],
        max_time_step=env["max_time_step"],
        grid_map_size=grid["size"],
        driver_type=driver_type,
        num_drivers=drv["num"],
        vel_drivers=drv["vel"],
        tolerance_percentage=tolerance_percentage,
        max_capacity=max_capacity,
        bag_capacity=bag_capacity,
        establishment_type=establishment_type,
        num_establishments=est["num"],
        prepare_time=prepare_time,
        operating_radius=operating_radius,
        production_capacity=production_capacity,
        percentage_allocation_driver=percentage_allocation_driver,
        use_estimate=use_estimate,
    )
