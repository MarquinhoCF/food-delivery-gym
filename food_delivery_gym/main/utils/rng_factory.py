import copy
from typing import Optional

import numpy as np


class RngFactory:
    """
    Subfluxos de RNG independentes por ator, restritos a um ambiente de simulação.

    Utiliza `numpy.random.SeedSequence.spawn` para que clones possam copiar a fábrica
    (o fluxo de geração restante) sem compartilhar objetos `Generator` com a origem.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        seed_sequence: Optional[np.random.SeedSequence] = None,
    ):
        if seed_sequence is not None:
            self._seed_sequence = seed_sequence
        else:
            self._seed_sequence = np.random.SeedSequence(seed)

    def next(self) -> np.random.Generator:
        child = self._seed_sequence.spawn(1)[0]
        return np.random.default_rng(child)

    def clone(self) -> "RngFactory":
        return RngFactory(seed_sequence=copy.deepcopy(self._seed_sequence))
