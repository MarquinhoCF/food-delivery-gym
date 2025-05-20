from food_delivery_gym.main.events.event import Event


class OrderEvent(Event):
    def __init__(self, order, customer_id, establishment_id, time, event_type):
        super().__init__(time, event_type)
        self.order = order
        self.customer_id = customer_id
        self.establishment_id = establishment_id

    # def __lt__(self, other):
    #     return self.creation_date < other.creation_date
