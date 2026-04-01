from food_delivery_gym.main.statistic.metrics._map_actor_bar_metric import _MapActorBarMetric

class EstablishmentOrdersFulfilledMetric(_MapActorBarMetric):
    _metric_key        = "orders_fulfilled"
    _bar_color         = "skyblue"
    _xlabel            = "Pedidos Atendidos"
    _ylabel_single     = "Pedidos Atendidos"
    _ylabel_agg        = "Pedidos Atendidos"
    _title_single      = "Pedidos Atendidos por Estabelecimento"
    _title_agg         = "Pedidos Atendidos por Estabelecimento (Estatísticas)"
    _map_actor_label   = "Estabelecimento"


class EstablishmentMaxOrdersInQueueMetric(_MapActorBarMetric):
    _metric_key        = "max_orders_in_queue"
    _bar_color         = "orange"
    _xlabel            = "Máx. Pedidos na Fila"
    _ylabel_single     = "Máx. Pedidos na Fila"
    _ylabel_agg        = "Máx. Pedidos na Fila"
    _title_single      = "Máximo de Pedidos na Fila por Estabelecimento"
    _title_agg         = "Máx. Pedidos na Fila por Estabelecimento (Estatísticas)"
    _map_actor_label   = "Estabelecimento"

class EstablishmentActiveTimeMetric(_MapActorBarMetric):
    _metric_key        = "active_time"
    _bar_color         = "purple"
    _xlabel            = "Tempo Ativo"
    _ylabel_single     = "Tempo Ativo"
    _ylabel_agg        = "Tempo Ativo"
    _title_single      = "Tempo Ativo por Estabelecimento"
    _title_agg         = "Tempo Ativo por Estabelecimento (Estatísticas)"
    _map_actor_label   = "Estabelecimento"