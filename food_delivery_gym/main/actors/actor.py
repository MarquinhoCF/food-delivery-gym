from abc import ABC
from typing import Optional, Any

from simpy import Process
from simpy.core import SimTime
from simpy.events import ProcessGenerator, Timeout

from food_delivery_gym.main.actors.resume import ResumeCursor, await_timeout
from food_delivery_gym.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv


class Actor(ABC):

    def __init__(self, environment: FoodDeliverySimpyEnv) -> None:
        self._environment = environment
        if environment.rng_factory is None:
            raise RuntimeError("FoodDeliverySimpyEnv requer rng_factory para criar atores")
        self.rng = environment.rng_factory.next()

    def publish_event(self, event) -> None:
        self._environment.add_event(event)

    def process(self, generator: ProcessGenerator) -> Process:
        return self._environment.process(generator)

    def timeout(self, delay: SimTime = 0, value: Optional[Any] = None) -> Timeout:
        return self._environment.timeout(delay=delay, value=value)

    def _await_timeout(self, phase: str, delay, resume: Optional[ResumeCursor] = None) -> ProcessGenerator:
        """
        Espera nomeada para processos clonáveis. Recebe o prefixo `_` para que `capture_wakes`
        preserve o nome do processo externo quando este for chamado via `yield from`.

        Passe um callable sem argumentos para atrasos baseados em RNG, de modo que não sejam
        recalculados enquanto o cursor ainda tiver tempo ``remaining`` (restante).
        """
        yield from await_timeout(self.timeout, phase, delay, resume)

    def _resume_remaining(self, resume: Optional[ResumeCursor]) -> ProcessGenerator:
        """
        Conclui uma espera de cauda (*loop poll*).
        Se está resumindo, espera o tempo restante e reentra no corpo normal.
        Se não está resumindo, não faz nada.
        """
        if resume is not None and resume.has_pending_remaining() and resume.phase is None:
            yield self.timeout(resume.delay(0))

    @property
    def now(self) -> SimTime:
        return self._environment.now

    @property
    def environment(self) -> FoodDeliverySimpyEnv:
        return self._environment
