"""Silêncio de ruído de eval — importar ANTES de catalog/SB3/gym nos workers."""

from __future__ import annotations

import os
import sys
import types
import warnings

_STDERR_FILTER_INSTALLED = False

_GYM_NOTICE_MARKERS = (
    "Gym has been unmaintained since 2022",
    "Please upgrade to Gymnasium, the maintained drop-in replacement of Gym",
    "Users of this version of Gym should be able to simply replace",
    "See the migration guide at https://gymnasium.farama.org",
)


class _GymNoticeStderrFilter:
    """Filtra o aviso multi-linha do Gym impresso via print(..., file=sys.stderr)."""

    def __init__(self, stream):
        self._stream = stream
        self._buf = ""

    def write(self, s):
        if not isinstance(s, str):
            s = s.decode("utf-8", errors="replace")
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if not any(marker in line for marker in _GYM_NOTICE_MARKERS):
                self._stream.write(line + "\n")

    def flush(self):
        if self._buf:
            if not any(marker in self._buf for marker in _GYM_NOTICE_MARKERS):
                self._stream.write(self._buf)
            self._buf = ""
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def in_worker_process() -> bool:
    """True se este processo não é o MainProcess do multiprocessing."""
    try:
        from multiprocessing import current_process

        return current_process().name != "MainProcess"
    except Exception:
        return False


def silence_eval_noise(*, filter_stderr: bool | None = None) -> None:
    """
    Reduz spam de TensorFlow/oneDNN e do aviso legado do Gym.

    Nos workers (spawn/forkserver) o __main__ reimporta scripts que puxam
    gym antes do silence antigo: chame isto no topo do script e use
    filter_stderr=True (padrão em workers) para cobrir o print do Gym.
    """
    global _STDERR_FILTER_INSTALLED

    if filter_stderr is None:
        filter_stderr = in_worker_process()

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    notices_mod = sys.modules.get("gym_notices.notices")
    if notices_mod is None:
        pkg = types.ModuleType("gym_notices")
        pkg.__path__ = []  # type: ignore[attr-defined]
        notices_mod = types.ModuleType("gym_notices.notices")
        notices_mod.notices = {}  # type: ignore[attr-defined]
        sys.modules["gym_notices"] = pkg
        sys.modules["gym_notices.notices"] = notices_mod
    else:
        notices = getattr(notices_mod, "notices", None)
        if isinstance(notices, dict):
            notices.clear()

    warnings.filterwarnings(
        "ignore",
        message=r".*Gym has been unmaintained since 2022.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*The environment .* is out of date.*",
    )

    if filter_stderr and not _STDERR_FILTER_INSTALLED:
        if not isinstance(sys.stderr, _GymNoticeStderrFilter):
            sys.stderr = _GymNoticeStderrFilter(sys.stderr)  # type: ignore[assignment]
        _STDERR_FILTER_INSTALLED = True
