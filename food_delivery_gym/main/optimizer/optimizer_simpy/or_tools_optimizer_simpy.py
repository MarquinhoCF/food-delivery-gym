from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from food_delivery_gym.main.cost.cost_function import CostFunction
from food_delivery_gym.main.optimizer.optimizer_simpy.optimizer_simpy import OptimizerSimpy
from food_delivery_gym.main.order.optimization_delivery_rejection import OptimizationDeliveryRejection
from food_delivery_gym.main.route.delivery_route_segment import DeliveryRouteSegment
from food_delivery_gym.main.route.pickup_route_segment import PickupRouteSegment
from food_delivery_gym.main.route.route import Route


class OrToolsOptimizerSimpy(OptimizerSimpy):
    def __init__(self, cost_function: CostFunction):
        super().__init__(cost_function)

    def select_driver(self, env, route):
        pass

    def process_orders(self, env, orders, rejected=False):
        if len(orders) == 0:
            return

        pickup_segments = list(map(lambda order: PickupRouteSegment(order), orders))
        delivery_segments = list(map(lambda order: DeliveryRouteSegment(order), orders))
        drivers = list(filter(lambda d: d.available, env.state.drivers))

        num_pickups = len(pickup_segments)
        num_deliveries = len(delivery_segments)
        num_drivers = len(drivers)
        all_locations = pickup_segments + delivery_segments + drivers
        num_locations = len(all_locations)

        # Matriz de distâncias
        distance_matrix = []
        for from_node in all_locations:
            row = []
            for to_node in all_locations:
                row.append(env.map.distance(from_node.coordinate, to_node.coordinate))
            distance_matrix.append(row)

        # Índices de estabelecimentos, clientes e motoristas
        pickup_indices = list(range(num_pickups))
        delivery_indices = list(range(num_pickups, num_pickups + num_deliveries))
        driver_indices = list(range(num_pickups + num_deliveries, num_locations))

        # Modelo do roteamento
        manager = pywrapcp.RoutingIndexManager(num_locations, num_drivers, driver_indices, driver_indices)
        routing = pywrapcp.RoutingModel(manager)

        # Função de custo (distância)
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return distance_matrix[from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        dimension_name = "distance"
        routing.AddDimension(
            transit_callback_index,
            0,  # no slack
            2000,  # vehicle maximum travel distance
            True,  # start cumul to zero
            dimension_name,
        )
        distance_dimension = routing.GetDimensionOrDie(dimension_name)

        # Restrições de coleta e entrega
        for pickup_index, delivery_index in zip(pickup_indices, delivery_indices):
            pickup_node = manager.NodeToIndex(pickup_index)
            delivery_node = manager.NodeToIndex(delivery_index)
            routing.AddPickupAndDelivery(pickup_node, delivery_node)
            routing.solver().Add(
                routing.VehicleVar(pickup_node) == routing.VehicleVar(delivery_node)
            )
            routing.solver().Add(
                distance_dimension.CumulVar(pickup_node) <= distance_dimension.CumulVar(delivery_node)
            )

        # # Tempo máximo para cada rota (opcional)
        # dimension_name = "time"
        # routing.AddDimension(
        #     transit_callback_index,
        #     0,  # Sem tempo de espera
        #     30,  # Tempo máximo por rota
        #     True,  # Tempo acumulado
        #     dimension_name
        # )
        # time_dimension = routing.GetDimensionOrDie(dimension_name)

        # # Restrições de tempo nas entregas e coletas
        # for node in range(num_locations):
        #     index = manager.NodeToIndex(node)
        #     time_dimension.CumulVar(index).SetRange(0, 30)

        # Configuração da pesquisa
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        # search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC)
        search_parameters.time_limit.FromSeconds(30)

        # Solução
        solution = routing.SolveWithParameters(search_parameters)

        # Impressão da solução
        if solution:
            for vehicle_id in range(num_drivers):
                index = routing.Start(vehicle_id)
                plan_output = 'Rota do motorista {}:\n'.format(vehicle_id)
                route_distance = 0
                first = True
                segments = []
                driver = None
                while not routing.IsEnd(index):
                    node_index = manager.IndexToNode(index)
                    # print(all_locations[node_index])
                    plan_output += ' {} ->'.format(node_index)
                    previous_index = index
                    index = solution.Value(routing.NextVar(index))
                    route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)

                    if first:
                        driver = all_locations[node_index]
                    else:
                        segments.append(all_locations[node_index])

                    first = False

                plan_output += ' {}\n'.format(manager.IndexToNode(index))
                plan_output += 'Distância da rota: {}\n'.format(route_distance)
                # print(plan_output)

                if driver is not None and len(segments) > 0:
                    route = Route(env, segments)
                    driver.receive_route_requests(route)
        else:
            print('Nenhuma solução encontrada!')
            for order in orders:
                event = OptimizationDeliveryRejection(
                    order_id=order.order_id,
                    customer_id=order.customer.customer_id,
                    establishment_id=order.establishment.establishment_id,
                    time=self.env.now
                )
                self._environment.add_event(event)
                env.add_rejected_delivery(order, OptimizationDeliveryRejection(env.now),event)
