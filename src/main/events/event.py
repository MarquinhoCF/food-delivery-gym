class Event:
    _id_counter = 0  # Variável estática para o ID

    def __init__(self, time, event_type):
        self.event_id = Event._generate_id()
        self.time = time
        self.event_type = event_type

    @classmethod
    def _generate_id(cls):
        cls._id_counter += 1
        return cls._id_counter

    # def __lt__(self, other):
    #     return self.creation_date < other.creation_date
