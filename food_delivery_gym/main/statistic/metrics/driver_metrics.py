from food_delivery_gym.main.statistic.metrics._map_actor_bar_metric import _MapActorBarMetric

class DriverTimeSpentOnDelivery(_MapActorBarMetric):
    _metric_key        = "time_spent_on_delivery"
    _bar_color         = "blue"
    _xlabel            = "Tempo Gasto na Entrega"
    _ylabel_single     = "Tempo Gasto na Entrega"
    _ylabel_agg        = "Tempo Gasto na Entrega"
    _title_single      = "Tempo Gasto na Entrega por Motorista"
    _title_agg         = "Tempo Gasto na Entrega por Motorista (Estatísticas)"
    _map_actor_label   = "Motoristas"


class DriverOrdersDeliveredMetric(_MapActorBarMetric):
    _metric_key        = "orders_delivered"
    _bar_color         = "blue"
    _xlabel            = "Pedidos Entregues"
    _ylabel_single     = "Pedidos Entregues"
    _ylabel_agg        = "Pedidos Entregues"
    _title_single      = "Pedidos Entregues por Motorista"
    _title_agg         = "Pedidos Entregues por Motorista (Estatísticas)"
    _map_actor_label   = "Motoristas"


class DriverTotalDistanceMetric(_MapActorBarMetric):
    _metric_key        = "total_distance"
    _bar_color         = "red"
    _xlabel            = "Distância Total Percorrida"
    _ylabel_single     = "Distância Total"
    _ylabel_agg        = "Distância Total"
    _title_single      = "Distância Total Percorrida por Motorista"
    _title_agg         = "Distância Total Percorrida por Motorista (Estatísticas)"
    _map_actor_label   = "Motoristas"


class DriverIdleTimeMetric(_MapActorBarMetric):
    _metric_key        = "idle_time"
    _bar_color         = "green"
    _xlabel            = "Tempo Ocioso"
    _ylabel_single     = "Tempo Ocioso"
    _ylabel_agg        = "Tempo Ocioso"
    _title_single      = "Tempo Ocioso por Motorista"
    _title_agg         = "Tempo Ocioso por Motorista (Estatísticas)"
    _map_actor_label   = "Motoristas"


class DriverTimeWaitingForOrderMetric(_MapActorBarMetric):
    _metric_key        = "time_waiting_for_order"
    _bar_color         = "teal"
    _xlabel            = "Tempo Esperando pelo Pedido"
    _ylabel_single     = "Tempo Esperando"
    _ylabel_agg        = "Tempo Esperando"
    _title_single      = "Tempo Esperando pelo Pedido por Motorista"
    _title_agg         = "Tempo Esperando pelo Pedido por Motorista (Estatísticas)"
    _map_actor_label   = "Motoristas"
