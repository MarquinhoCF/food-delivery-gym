from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

from simpy.core import SimTime
from simpy.events import ProcessGenerator

DelaySpec = Union[SimTime, Callable[[], SimTime]]


@dataclass
class ResumeCursor:
    """
    Cursor que descreve onde um processo clonado deve continuar.

    * `phase is None`-> novo início (executar todas as fases desde o início) ou resumo apenas com tempo restante para loops de espera de cauda.
    * `phase == "..."` —> pular fases antes desse nome; o primeiro `enter` que corresponder usa `remaining` como o atraso do timeout.
    """

    phase: Optional[str] = None
    remaining: Optional[SimTime] = None
    extras: dict[str, Any] = field(default_factory=dict)

    _passed_resume_point: bool = field(default=False, init=False, repr=False)
    _remaining_consumed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        # Novo início, ou resumo apenas com tempo restante (loops de espera de cauda): todo `enter()` é executado.
        if self.phase is None:
            self._passed_resume_point = True

    @property
    def at_start(self) -> bool:
        return self.phase is None and self.remaining is None

    @property
    def is_resuming(self) -> bool:
        return not self.at_start

    def enter(self, phase: str) -> bool:
        """Retorna True se a espera desta fase e os efeitos colaterais subsequentes devem ser executados."""
        if not self._passed_resume_point:
            if phase != self.phase:
                return False
            self._passed_resume_point = True
            return True
        return True

    def delay(self, full: SimTime) -> SimTime:
        """Usa `remaining` uma vez na fase de resumo, então atrasos completos."""
        if self.remaining is not None and not self._remaining_consumed:
            self._remaining_consumed = True
            value = self.remaining
            self.remaining = None
            return value
        return full

    def has_pending_remaining(self) -> bool:
        return self.remaining is not None and not self._remaining_consumed


def _resolve_delay(delay: DelaySpec) -> SimTime:
    return delay() if callable(delay) else delay


def await_timeout(
    timeout_fn,
    phase: str,
    delay: DelaySpec,
    resume: Optional[ResumeCursor] = None,
) -> ProcessGenerator:
    """
    Espera nomeada utilizável por atores e geradores não-atores.

    `delay` pode ser um callable para que as durações baseadas em RNG não sejam recalculadas na fase de resumo
    quando `remaining` é usado em vez disso.
    """
    cursor = resume or ResumeCursor()
    if not cursor.enter(phase):
        return
    if cursor.has_pending_remaining():
        yield timeout_fn(cursor.delay(0))
    else:
        yield timeout_fn(cursor.delay(_resolve_delay(delay)))
