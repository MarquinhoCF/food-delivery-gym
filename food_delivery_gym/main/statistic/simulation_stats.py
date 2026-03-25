"""
Utilitário de estatísticas para simulações do food-delivery-gym.

Uso básico
──────────
    stats = SimulationStats()

    for i in range(num_runs):
        ... rodar episódio ...
        stats.register_episode(simpy_env, reward=r, length=l,
                               truncated=t, orders_generated=n)

    stats.finalize()
    stats.save(dir_path="resultados/", fmt="json")

Formato de armazenamento padrão: NPZ comprimido (np.savez_compressed)
──────────────────────────────────────────────────────────────────────
Estrutura de chaves — separador '__', tudo float64, None → NaN:

  Séries por episódio (arrays de tamanho N):
    ep__rewards / ep__lengths / ep__simpy_times / ep__truncated
    ep__orders_generated / ep__delivery_time / ep__distance

  Métricas por agente (arrays de tamanho N):
    driver__<id>__<metric>
    establishment__<id>__<metric>

  Estatísticas agregadas (arrays de tamanho 1):
    agg__<metric>__avg / __std_dev / __median / __n

Formato JSON (conversão via npz_to_json / json_to_npz)
────────────────────────────────────────────────────────
{
  "sim": [
    {
      "reward": 42.0, "episode_length": 100, ...
      "driver":        { "0": { "idle_time": 600.0, ... } },
      "establishment": { "0": { "orders_fulfilled": 48, ... } }
    }, ...
  ],
  "aggregate": {
    "rewards": { "avg": 40.0, "std_dev": 2.0, "median": 41.0, "n": 20 }, ...
  }
}

Funções de conversão NPZ ↔ JSON (módulo-nível):
    npz_to_json(npz_path, json_path)
    json_to_npz(json_path, npz_path)
"""

from __future__ import annotations

import json
import math
import os
import statistics as stt
import traceback
from typing import IO, Literal

import numpy as np


# ════════════════════════════════════════════════════════════════════════
#  Funções utilitárias de conversão (módulo-nível)
# ════════════════════════════════════════════════════════════════════════

def npz_to_json(npz_path: str, json_path: str) -> None:
    stats = SimulationStats.load(npz_path)
    _write_json(stats, json_path)
    print(f"Convertido: {npz_path} → {json_path}")


def json_to_npz(json_path: str, npz_path: str) -> None:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    arrays = _json_data_to_npz_arrays(data)
    np.savez_compressed(npz_path, **arrays)
    print(f"Convertido: {json_path} → {npz_path}")


# ════════════════════════════════════════════════════════════════════════
#  Helpers internos
# ════════════════════════════════════════════════════════════════════════

def _to_f64(values) -> np.ndarray:
    if values is None:
        return np.array([], dtype=np.float64)
    return np.array(
        [np.nan if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v)
         for v in values],
        dtype=np.float64,
    )


def _json_default(obj):
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Tipo não serializável: {type(obj)}")


def _write_json(stats: SimulationStats, path: str) -> None:
    data = {"sim": stats.sim, "aggregate": stats.aggregate}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)


def _json_data_to_npz_arrays(data: dict) -> dict[str, np.ndarray]:
    sim_list  = data.get("sim", [])
    aggregate = data.get("aggregate", {})
    n         = len(sim_list)
    arrays: dict[str, np.ndarray] = {}

    _EP_KEYS = [
        ("reward",           "ep__rewards"),
        ("episode_length",   "ep__lengths"),
        ("simpy_final_time", "ep__simpy_times"),
        ("truncated",        "ep__truncated"),
        ("orders_generated", "ep__orders_generated"),
        ("delivery_time",    "ep__delivery_time"),
        ("distance",         "ep__distance"),
    ]
    for sim_key, npz_key in _EP_KEYS:
        arrays[npz_key] = _to_f64([ep.get(sim_key) for ep in sim_list])

    if n > 0:
        for did, metrics in sim_list[0].get("driver", {}).items():
            for metric in metrics:
                arrays[f"driver__{did}__{metric}"] = _to_f64(
                    [ep["driver"][did].get(metric) for ep in sim_list]
                )
        for eid, metrics in sim_list[0].get("establishment", {}).items():
            for metric in metrics:
                arrays[f"establishment__{eid}__{metric}"] = _to_f64(
                    [ep["establishment"][eid].get(metric) for ep in sim_list]
                )

    for metric, stats in aggregate.items():
        if not stats:
            continue
        for stat_name, val in stats.items():
            if val is None:
                continue
            try:
                dtype = np.int64 if stat_name == "n" else np.float64
                arrays[f"agg__{metric}__{stat_name}"] = np.array([val], dtype=dtype)
            except (TypeError, ValueError):
                pass

    return arrays


# ════════════════════════════════════════════════════════════════════════
#  Classe principal
# ════════════════════════════════════════════════════════════════════════

class SimulationStats:
    """
    Ponto único de coleta, agregação e persistência de estatísticas.

    Ciclo de vida
    ─────────────
    1. stats = SimulationStats()
    2. for each episode:  stats.register_episode(simpy_env, reward, ...)
    3. stats.finalize()          ← computa agregados
    4. stats.save(dir_path)      ← persiste NPZ ou JSON
    5. stats.write_report(file)  ← relatório em texto

    Após finalize(), acesso:
        stats.episodes["rewards"]            # lista de N recompensas
        stats.drivers["0"]["distance"]       # lista de N distâncias do driver 0
        stats.aggregate["rewards"]["avg"]
        stats.sim[i]["reward"]               # visão por episódio (lazy)
        stats.sim[i]["driver"]["0"]["distance"]
    """

    def __init__(self) -> None:
        self._raw_episodes: list[dict] = []

        # Populados por finalize()
        self.episodes: dict       = {}
        self.drivers: dict        = {}
        self.establishments: dict = {}
        self.aggregate: dict      = {}
        self._num_runs: int       = 0
        self._sim: list[dict] | None = None

    # ════════════════════════════════════════════════════════════════════
    #  API principal
    # ════════════════════════════════════════════════════════════════════

    def register_episode(
        self,
        simpy_env,
        reward: float,
        length: int,
        truncated: bool,
        orders_generated: int,
    ) -> None:
        """
        Registra os dados de um episódio completo.

        Chame uma vez por episódio, antes de reset(). Extrai os dados
        diretamente dos drivers e establishments via get_episode_stats().
        """
        ep: dict = {
            "reward":           float(reward),
            "length":           int(length),
            "simpy_time":       float(simpy_env.now),
            "truncated":        bool(truncated),
            "orders_generated": int(orders_generated),
            "drivers": {
                str(d.driver_id): d.get_episode_stats()
                for d in simpy_env.state.drivers
            },
            "establishments": {
                str(e.establishment_id): e.get_episode_stats()
                for e in simpy_env.state.establishments
            },
        }
        self._raw_episodes.append(ep)
        self._sim = None  # invalida cache lazy

    def finalize(self) -> "SimulationStats":
        """
        Processa todos os episódios registrados e computa os agregados.

        Chame após todos os register_episode() e antes de save() ou
        write_report(). Retorna self para encadeamento.
        """
        eps = self._raw_episodes
        n   = len(eps)
        self._num_runs = n
        self._sim      = None

        if n == 0:
            return self

        # ── Séries de episódio ────────────────────────────────────────
        rewards          = [ep["reward"]           for ep in eps]
        lengths          = [ep["length"]           for ep in eps]
        simpy_times      = [ep["simpy_time"]       for ep in eps]
        truncated        = [ep["truncated"]        for ep in eps]
        orders_generated = [ep["orders_generated"] for ep in eps]

        # ── Métricas por driver ───────────────────────────────────────
        self.drivers = {}
        driver_ids = list(eps[0]["drivers"].keys())

        for did in driver_ids:
            self.drivers[did] = {}
            sample = eps[0]["drivers"].get(did, {})

            for metric, val in sample.items():
                if isinstance(val, dict):
                    # Métrica aninhada (ex.: reordering stats do DynamicRouteDriver)
                    for sub_key in val:
                        col = f"{metric}_{sub_key}"
                        self.drivers[did][col] = [
                            ep["drivers"].get(did, {}).get(metric, {}).get(sub_key)
                            for ep in eps
                        ]
                else:
                    self.drivers[did][metric] = [
                        ep["drivers"].get(did, {}).get(metric)
                        for ep in eps
                    ]

        # ── Métricas por estabelecimento ──────────────────────────────
        self.establishments = {}
        est_ids = list(eps[0]["establishments"].keys())

        for eid in est_ids:
            self.establishments[eid] = {}
            sample = eps[0]["establishments"].get(eid, {})
            for metric in sample:
                self.establishments[eid][metric] = [
                    ep["establishments"].get(eid, {}).get(metric)
                    for ep in eps
                ]

        # ── Séries derivadas ─────────────────────────────────────────
        delivery_time = [
            sum(
                ep["drivers"].get(did, {}).get("time_spent_on_delivery", 0) or 0
                for did in driver_ids
            )
            for ep in eps
        ]

        # Distância exclui episódios truncados (métrica inválida nesses casos)
        distance = [
            None if ep["truncated"] else
            sum(
                ep["drivers"].get(did, {}).get("total_distance", 0) or 0
                for did in driver_ids
            )
            for ep in eps
        ]

        self.episodes = {
            "rewards":          rewards,
            "lengths":          lengths,
            "simpy_times":      simpy_times,
            "truncated":        [bool(t) for t in truncated],
            "orders_generated": orders_generated,
            "delivery_time":    delivery_time,
            "distance":         distance,
        }

        # ── Agregados ─────────────────────────────────────────────────
        self.aggregate = {
            "rewards":       self._safe_stats(rewards),
            "lengths":       self._safe_stats(lengths),
            "simpy_times":   self._safe_stats([v for v in simpy_times if v is not None]),
            "orders":        self._safe_stats(orders_generated),
            "delivery_time": self._safe_stats(delivery_time),
            "distance":      self._safe_stats([v for v in distance if v is not None]),
        }

        return self

    # ════════════════════════════════════════════════════════════════════
    #  Visão por episódio (lazy)
    # ════════════════════════════════════════════════════════════════════

    @property
    def sim(self) -> list[dict]:
        if self._sim is None:
            self._sim = self._build_sim_list()
        return self._sim

    def _build_sim_list(self) -> list[dict]:
        ep = self.episodes
        return [
            {
                "reward":           ep["rewards"][i],
                "episode_length":   ep["lengths"][i],
                "simpy_final_time": ep["simpy_times"][i],
                "truncated":        ep["truncated"][i],
                "orders_generated": ep["orders_generated"][i],
                "delivery_time":    ep["delivery_time"][i],
                "distance":         ep["distance"][i],
                "driver": {
                    did: {m: vals[i] if i < len(vals) else None
                          for m, vals in metrics.items()}
                    for did, metrics in self.drivers.items()
                },
                "establishment": {
                    eid: {m: vals[i] if i < len(vals) else None
                          for m, vals in metrics.items()}
                    for eid, metrics in self.establishments.items()
                },
            }
            for i in range(self._num_runs)
        ]

    # ════════════════════════════════════════════════════════════════════
    #  Persistência — save / load
    # ════════════════════════════════════════════════════════════════════

    def save(
        self,
        dir_path: str,
        fmt: Literal["npz", "json"] = "npz",
        file_name: str | None = None,
    ) -> None:
        if fmt not in ("npz", "json"):
            raise ValueError(f"Formato desconhecido: '{fmt}'. Use 'npz' ou 'json'.")

        os.makedirs(dir_path, exist_ok=True)
        name = file_name or f"metrics_data.{fmt}"
        path = os.path.join(dir_path, name)

        try:
            if fmt == "npz":
                np.savez_compressed(path, **self._build_npz_arrays())
                print(f"Métricas salvas em {path}")
            else:
                _write_json(self, path)
                print(f"Métricas salvas em {path}")
        except Exception as e:
            print(f"Erro ao salvar métricas em {path}: {e}")
            traceback.print_exc()

    @staticmethod
    def load(file_path: str) -> "SimulationStats":
        raw    = np.load(file_path, allow_pickle=False)
        result = SimulationStats.__new__(SimulationStats)
        result._sim              = None
        result._raw_episodes     = []
        result.episodes          = {}
        result.drivers           = {}
        result.establishments    = {}
        result.aggregate         = {}

        _EP_MAP = {
            "ep__rewards":          "rewards",
            "ep__lengths":          "lengths",
            "ep__simpy_times":      "simpy_times",
            "ep__truncated":        "truncated",
            "ep__orders_generated": "orders_generated",
            "ep__delivery_time":    "delivery_time",
            "ep__distance":         "distance",
        }

        for key in raw.files:
            arr   = raw[key]
            parts = key.split("__")

            if key in _EP_MAP:
                result.episodes[_EP_MAP[key]] = arr
            elif parts[0] == "driver" and len(parts) == 3:
                result.drivers.setdefault(parts[1], {})[parts[2]] = arr
            elif parts[0] == "establishment" and len(parts) == 3:
                result.establishments.setdefault(parts[1], {})[parts[2]] = arr
            elif parts[0] == "agg" and len(parts) == 3:
                metric, stat = parts[1], parts[2]
                result.aggregate.setdefault(metric, {})[stat] = (
                    int(arr[0]) if stat == "n" else float(arr[0])
                )

        result._num_runs = (
            len(next(iter(result.episodes.values())))
            if result.episodes else 0
        )
        return result

    # ════════════════════════════════════════════════════════════════════
    #  Relatório em texto
    # ════════════════════════════════════════════════════════════════════

    def write_report(self, results_file: IO, num_truncated: int = 0) -> None:
        self._write_block(results_file, "Estatísticas das Recompensas",
                          self.aggregate.get("rewards"))
        self._write_block(results_file, "Estatísticas do Tamanho dos Episódios (passos)",
                          self.aggregate.get("lengths"))
        self._write_block(results_file, "Estatísticas do Tempo de Simulação SimPy (último passo)",
                          self.aggregate.get("simpy_times"))
        self._write_block(results_file, "Estatísticas do Tempo Gasto com Entregas",
                          self.aggregate.get("delivery_time"))

        num_valid = self._num_runs - num_truncated
        results_file.write(
            f"\n---> Execuções válidas para métricas de distância: {num_valid}/{self._num_runs}\n"
        )
        if num_truncated > 0:
            results_file.write(
                f"* {num_truncated} execução(ões) truncada(s) excluída(s) das métricas de distância\n"
            )
        self._write_block(results_file, "Estatísticas da Distância Percorrida",
                          self.aggregate.get("distance"))
        self._write_block(results_file, "Estatísticas do Número de Pedidos Gerados",
                          self.aggregate.get("orders"))

    # ════════════════════════════════════════════════════════════════════
    #  Helpers
    # ════════════════════════════════════════════════════════════════════

    @staticmethod
    def _safe_stats(values: list) -> dict | None:
        clean = [float(v) for v in values if v is not None]
        if not clean:
            return None
        stats: dict = {
            "avg":     sum(clean) / len(clean),
            "std_dev": stt.stdev(clean) if len(clean) > 1 else 0.0,
            "median":  stt.median(clean),
            "mode":    None,
            "n":       len(clean),
        }
        try:
            stats["mode"] = stt.mode(clean)
        except stt.StatisticsError:
            stats["mode"] = "Sem moda única"
        return stats

    @staticmethod
    def _write_block(results_file: IO, title: str, stats: dict | None) -> None:
        results_file.write(f"\n---> {title}:\n")
        if stats is None:
            results_file.write("* NULO - Sem dados válidos\n")
            return
        results_file.write(f"* N amostras:    {stats['n']}\n")
        results_file.write(f"* Média:         {stats['avg']:.4f}\n")
        results_file.write(f"* Desvio Padrão: {stats['std_dev']:.4f}\n")
        results_file.write(f"* Mediana:       {stats['median']:.4f}\n")
        results_file.write(f"* Moda:          {stats['mode']}\n")

    def _build_npz_arrays(self) -> dict[str, np.ndarray]:
        arrays: dict[str, np.ndarray] = {}

        _EP_KEYS = [
            ("rewards",          "ep__rewards"),
            ("lengths",          "ep__lengths"),
            ("simpy_times",      "ep__simpy_times"),
            ("truncated",        "ep__truncated"),
            ("orders_generated", "ep__orders_generated"),
            ("delivery_time",    "ep__delivery_time"),
            ("distance",         "ep__distance"),
        ]
        for ep_key, npz_key in _EP_KEYS:
            arrays[npz_key] = _to_f64(self.episodes.get(ep_key, []))

        for driver_id, metrics in self.drivers.items():
            for metric, values in metrics.items():
                arrays[f"driver__{driver_id}__{metric}"] = _to_f64(values)

        for est_id, metrics in self.establishments.items():
            for metric, values in metrics.items():
                arrays[f"establishment__{est_id}__{metric}"] = _to_f64(values)

        for metric, stats in self.aggregate.items():
            if not stats:
                continue
            for stat_name, val in stats.items():
                if val is None:
                    continue
                try:
                    dtype = np.int64 if stat_name == "n" else np.float64
                    arrays[f"agg__{metric}__{stat_name}"] = np.array([val], dtype=dtype)
                except (TypeError, ValueError):
                    pass

        return arrays