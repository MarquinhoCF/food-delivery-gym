from abc import ABC, abstractmethod


class Metric(ABC):

    @abstractmethod
    def view(self, ax) -> None:
        pass
