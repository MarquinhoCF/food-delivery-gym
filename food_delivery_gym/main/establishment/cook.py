from simpy.core import SimTime

from food_delivery_gym.main.actors.map_actor import MapActor
from food_delivery_gym.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv
from food_delivery_gym.main.order.order import Order

class Cook():
    def __init__(self, enviroment: FoodDeliverySimpyEnv):
        self.enviroment = enviroment
        self.is_cooking = False
        self.orders_accepted: list[Order] = []
        self.overloaded_until: SimTime = 0
        self.current_order_duration: SimTime = 0
        self.order_list_duration: SimTime = 0

    def add_order_to_list(self, order: Order):
        self.orders_accepted.append(order)

    def get_length_orders_accepted(self):
        return len(self.orders_accepted)
    
    def pop_order(self):
        return self.orders_accepted.pop(0)
    
    def get_overloaded_until(self):
        return self.overloaded_until
    
    def set_current_order_duration(self, duration: SimTime):
        self.current_order_duration = duration
    
    def get_is_cooking(self):
        return self.is_cooking
    
    def set_is_cooking(self, is_cooking: bool):
        self.is_cooking = is_cooking
    
    def update_overload_time(self, estimated_time = None, after_establishment_accept_order = False) -> None:

        # Se uma estimativa é passada, ela é usada para calcular o tempo de ocupação
        if estimated_time is not None:

            # Se esse método for chamado dentro do método process_accepted_orders
            if after_establishment_accept_order:

                #   Garantimos que as durações sejam atualizadas somente se o restaurante estiver vazio e com a duração 
                # do pedido atual igual a 0, pois se a duração do pedido atual é diferente de 0, significa que esse é o 
                # primeiro pedido ou o primeiro pedido depois certo tempo vazio
                if self.current_order_duration == 0:

                    if self.order_list_duration != 0:
                        self.order_list_duration -= estimated_time

                    self.current_order_duration = estimated_time

                    self.overloaded_until = self.enviroment.now + self.current_order_duration + self.order_list_duration
                    
                else:
                    self.overloaded_until = max(self.overloaded_until, self.enviroment.now)

            #   Se esse método for chamado fora do método process_accepted_orders só irá atualizar a duração do pedido atual 
            # caso seja o primeiro pedido ou o primeiro pedido depois certo tempo vazio
            #   Além disso a estimativa só será adicionada na duração da lista de pedidos aceitos se for chamado fora do método
            # process_accepted_orders
            else:
                if self.order_list_duration == 0 and self.current_order_duration == 0:
                    self.current_order_duration = estimated_time
                    self.overloaded_until = self.enviroment.now + self.current_order_duration
                else:
                    self.order_list_duration += estimated_time
                    self.overloaded_until += estimated_time
        
        # Se nenhuma estimativa é passada, o tempo de sobrecarga é atualizado com o tempo atual
        else:
            self.overloaded_until = max(self.overloaded_until, self.enviroment.now)
