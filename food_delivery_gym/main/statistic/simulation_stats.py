"""
Utilitário de estatísticas para simulações do food-delivery-gym.

Formato de armazenamento padrão: NPZ comprimido (np.savez_compressed)
──────────────────────────────────────────────────────────────────────
Estrutura de chaves — separador '__', tudo float64, None → NaN:

  Séries por episódio (arrays de tamanho N):
    ep__rewards
    ep__lengths
    ep__simpy_times
    ep__truncated
    ep__orders_generated
    ep__delivery_time
    ep__distance

  Métricas por agente (arrays de tamanho N):
    driver__<id>__<metric>
    establishment__<id>__<metric>

  Estatísticas agregadas (arrays de tamanho 1):
    agg__<metric>__avg
    agg__<metric>__std_dev
    agg__<metric>__median
    agg__<metric>__n          ← int64

Formato JSON (conversão via npz_to_json / json_to_npz)
────────────────────────────────────────────────────────
Estrutura orientada por simulação — fácil de inspecionar:
{
  "sim": [
    {
      "reward": 42.0,
      "episode_length": 100,
      "simpy_final_time": 3600.0,
      "truncated": false,
      "orders_generated": 50,
      "delivery_time": 1200.0,
      "distance": 300.0,
      "driver":        { "0": { "idle_time": 600.0, ... } },
      "establishment": { "0": { "orders_fulfilled": 48, ... } }
    },
    ...
  ],
  "aggregate": {
    "rewards": { "avg": 40.0, "std_dev": 2.0, "median": 41.0, "mode": null, "n": 20 },
    ...
  }
}

Funções utilitárias de conversão (módulo-nível):
    npz_to_json(npz_path, json_path)   – converte NPZ → JSON intuitivo
    json_to_npz(json_path, npz_path)   – converte JSON → NPZ comprimido
"""

from __future__ import annotations

import json
import math
import os
import traceback
import statistics as stt
from typing import IO, Literal

import numpy as np


# ════════════════════════════════════════════════════════════════════════
#  Funções utilitárias de conversão (módulo-nível)
# ════════════════════════════════════════════════════════════════════════

def npz_to_json(npz_path: str, json_path: str) -> None:
    """
    Converte um arquivo NPZ salvo por SimulationStats.save() para JSON
    com a estrutura intuitiva orientada por simulação.

    Exemplo:
        npz_to_json("results/metrics_data.npz", "results/metrics_data.json")
    """
    stats = SimulationStats.load(npz_path)
    _write_json(stats, json_path)
    print(f"Convertido: {npz_path} → {json_path}")


def json_to_npz(json_path: str, npz_path: str) -> None:
    """
    Converte um JSON com estrutura intuitiva de volta para NPZ comprimido.

    Exemplo:
        json_to_npz("results/metrics_data.json", "results/metrics_data.npz")
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    arrays = _json_data_to_npz_arrays(data)
    np.savez_compressed(npz_path, **arrays)
    print(f"Convertido: {json_path} → {npz_path}")


# ════════════════════════════════════════════════════════════════════════
#  Helpers internos de conversão
# ════════════════════════════════════════════════════════════════════════

def _to_f64(values) -> np.ndarray:
    """Lista (com possíveis None/NaN) → float64, None/NaN → np.nan."""
    if values is None:
        return np.array([], dtype=np.float64)
    return np.array(
        [np.nan if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v)
         for v in values],
        dtype=np.float64,
    )


def _json_default(obj):
    """Serializa tipos NumPy e NaN para JSON."""
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
    """Serializa SimulationStats para o JSON intuitivo."""
    data = {"sim": stats.sim, "aggregate": stats.aggregate}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)


def _json_data_to_npz_arrays(data: dict) -> dict[str, np.ndarray]:
    """Reconstrói o dicionário de arrays NPZ a partir do JSON intuitivo."""
    sim_list  = data.get("sim", [])
    aggregate = data.get("aggregate", {})
    n         = len(sim_list)

    arrays: dict[str, np.ndarray] = {}

    # Séries de episódio
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

    # Métricas por driver e establishment
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

    # Agregados
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
    Agrega e organiza todas as estatísticas de um conjunto de simulações.

    Atributos públicos após construção
    ────────────────────────────────────
    episodes : dict
        Séries brutas como listas, uma por métrica de episódio.

    drivers : dict
        driver_id → {metric → list[float | None]}

    establishments : dict
        est_id → {metric → list[float | None]}

    aggregate : dict
        Estatísticas resumidas (avg / std_dev / median / mode / n).

    sim : list[dict]  (property, lazy)
        Visão orientada por simulação. Construída na primeira chamada.
        stats.sim[4]["reward"]
        stats.sim[4]["driver"]["0"]["distance"]
    """

    _DRIVER_KEY_MAP: dict[str, str] = {
        "orders_delivered":       "orders_delivered",
        "time_spent_on_delivery": "time_spent_delivery",
        "idle_time":              "idle_time",
        "time_waiting_for_order": "time_waiting_order",
        "total_distance":         "distance",
    }
    _EST_KEY_MAP: dict[str, str] = {
        "orders_fulfilled":    "orders_fulfilled",
        "idle_time":           "idle_time",
        "active_time":         "active_time",
        "max_orders_in_queue": "max_orders_queue",
    }

    # ════════════════════════════════════════════════════════════════════
    #  Construção
    # ════════════════════════════════════════════════════════════════════

    def __init__(
        self,
        num_runs: int,
        rewards: list[float],
        lengths: list[int],
        simpy_times: list[float | None],
        orders_generated: list[int],
        truncated: list[bool],
        driver_metrics_raw: dict,
        establishment_metrics_raw: dict,
        geral_statistics: dict | None = None,
    ) -> None:
        self._num_runs         = num_runs
        self._geral_statistics = geral_statistics
        self._sim: list[dict] | None = None

        # 1. Normaliza métricas por agente
        self.drivers        = self._parse_driver_metrics(driver_metrics_raw)
        self.establishments = self._parse_establishment_metrics(establishment_metrics_raw)

        # 2. Séries derivadas por episódio
        delivery_time = self._sum_per_episode(num_runs, self.drivers, "time_spent_delivery")
        distance      = self._sum_per_episode(
            num_runs, self.drivers, "distance", mask_truncated=truncated,
        )

        self.episodes: dict = {
            "rewards":          list(rewards),
            "lengths":          list(lengths),
            "simpy_times":      list(simpy_times),
            "truncated":        [bool(t) for t in truncated],
            "orders_generated": list(orders_generated),
            "delivery_time":    delivery_time,
            "distance":         distance,
        }

        # 3. Estatísticas agregadas
        self.aggregate: dict = {
            "rewards":       self._safe_stats(rewards),
            "lengths":       self._safe_stats(lengths),
            "simpy_times":   self._safe_stats([v for v in simpy_times if v is not None]),
            "orders":        self._safe_stats(orders_generated),
            "delivery_time": self._safe_stats(delivery_time),
            "distance":      self._safe_stats([v for v in distance if v is not None]),
        }

    # ════════════════════════════════════════════════════════════════════
    #  Visão orientada por simulação (lazy)
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
    #  Interface pública — save / load
    # ════════════════════════════════════════════════════════════════════

    def save(
        self,
        dir_path: str,
        fmt: Literal["npz", "json"] = "npz",
        file_name: str | None = None,
    ) -> None:
        """
        Salva as estatísticas em disco.

        Args:
            dir_path:  Diretório de destino.
            fmt:       "npz" (padrão, comprimido) ou "json" (legível).
            file_name: Nome do arquivo. Se None, usa "metrics_data.npz"
                       ou "metrics_data.json" conforme o formato.
        """
        if fmt not in ("npz", "json"):
            raise ValueError(f"Formato desconhecido: '{fmt}'. Use 'npz' ou 'json'.")

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
    def load(file_path: str) -> SimulationStats:
        """
        Carrega um NPZ salvo por save() e reconstrói a instância.

        Exemplos de acesso:
            stats = SimulationStats.load("results/metrics_data.npz")
            stats.episodes["rewards"]             # array de N recompensas
            stats.drivers["0"]["distance"]        # array de N distâncias
            stats.aggregate["rewards"]["avg"]
            stats.sim[4]["reward"]                # visão por simulação (lazy)
            stats.sim[4]["driver"]["0"]["distance"]
        """
        raw    = np.load(file_path, allow_pickle=False)
        result = SimulationStats.__new__(SimulationStats)
        result._geral_statistics = None
        result._sim              = None
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
    #  Interface pública — relatório
    # ════════════════════════════════════════════════════════════════════

    def write_report(self, results_file: IO, num_truncated: int = 0) -> None:
        """Escreve o relatório completo de estatísticas no arquivo aberto."""
        self._write_block(results_file, "Estatísticas das Recompensas",
                          self.aggregate["rewards"])
        self._write_block(results_file, "Estatísticas do Tamanho dos Episódios (passos)",
                          self.aggregate["lengths"])
        self._write_block(results_file, "Estatísticas do Tempo de Simulação SimPy (último passo)",
                          self.aggregate["simpy_times"])
        self._write_block(results_file, "Estatísticas do Tempo Gasto com Entregas",
                          self.aggregate["delivery_time"])

        num_valid = self._num_runs - num_truncated
        results_file.write(
            f"\n---> Execuções válidas para métricas de distância: {num_valid}/{self._num_runs}\n"
        )
        if num_truncated > 0:
            results_file.write(
                f"* {num_truncated} execução(ões) truncada(s) excluída(s) das métricas de distância\n"
            )
        self._write_block(results_file, "Estatísticas da Distância Percorrida",
                          self.aggregate["distance"])
        self._write_block(results_file, "Estatísticas do Número de Pedidos Gerados",
                          self.aggregate["orders"])

        if self._geral_statistics:
            results_file.write("\n---> Estatísticas Finais:\n")
            results_file.write(self._format_geral_statistics(self._geral_statistics))

    # ════════════════════════════════════════════════════════════════════
    #  Parsing de métricas brutas
    # ════════════════════════════════════════════════════════════════════

    def _parse_driver_metrics(self, raw: dict) -> dict:
        drivers = {}
        for driver_id, metrics in raw.items():
            data = {}
            for raw_key, values in metrics.items():
                if not values:
                    continue
                clean_key = self._DRIVER_KEY_MAP.get(raw_key, raw_key)
                if isinstance(values[0], dict):
                    for field in values[0].keys():
                        data[f"{clean_key}_{field}"] = [v.get(field) for v in values]
                else:
                    data[clean_key] = list(values)
            drivers[driver_id] = data
        return drivers

    def _parse_establishment_metrics(self, raw: dict) -> dict:
        return {
            est_id: {
                self._EST_KEY_MAP.get(key, key): list(values)
                for key, values in metrics.items()
            }
            for est_id, metrics in raw.items()
        }

    # ════════════════════════════════════════════════════════════════════
    #  Cálculos por episódio
    # ════════════════════════════════════════════════════════════════════

    def _sum_per_episode(
        self,
        num_runs: int,
        agents: dict,
        metric: str,
        mask_truncated: list[bool] | None = None,
    ) -> list[float | None]:
        totals = []
        for i in range(num_runs):
            if mask_truncated and mask_truncated[i]:
                totals.append(None)
                continue
            total = sum(
                agent[metric][i]
                for agent in agents.values()
                if i < len(agent.get(metric, []))
            )
            totals.append(total)
        return totals

    # ════════════════════════════════════════════════════════════════════
    #  Estatísticas descritivas
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

    # ════════════════════════════════════════════════════════════════════
    #  Escrita de relatório
    # ════════════════════════════════════════════════════════════════════

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

    @staticmethod
    def _format_geral_statistics(statistics: dict) -> str:
        lines = []
        for section in ("establishments", "drivers"):
            if section not in statistics:
                continue
            lines.append(f"{section.title()}:")
            for agent_id, stats in statistics[section].items():
                label = "Establishment" if section == "establishments" else "Driver"
                lines.append(f"  {label} {agent_id}:")
                for key, value in stats.items():
                    lines.append(f"    {key.replace('_', ' ').title()}:")
                    for stat, stat_value in value.items():
                        lines.append(f"      {stat.title()}: {stat_value}")
                lines.append("")
        return "\n".join(lines)

    # ════════════════════════════════════════════════════════════════════
    #  Serialização NPZ
    # ════════════════════════════════════════════════════════════════════

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