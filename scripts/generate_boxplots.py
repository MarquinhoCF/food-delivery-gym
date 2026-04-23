from __future__ import annotations

import argparse
import json
import os
import warnings
from typing import Optional

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

DEFAULT_RESULTS_DIR = "./data/runs/execucoes"
DEFAULT_OUTPUT_DIR  = "./data/runs/figuras"
DEFAULT_OBJECTIVE   = 3

KNOWN_AGENTS: dict[str, str] = {
    "random":                       "Aleatório",
    "first_driver":                 "Primeiro Mot.",
    "nearest_driver":               "Mot. Próximo",
    "lowest_route_cost":            "Menor Custo",
    "lowest_marginal_route_cost":   "Menor Custo Marg.",
    "ppo_18M_steps":                "PPO Padrão",
    "ppo_18M_steps_otimizado":      "PPO Otimizado",
}

HEURISTIC_DIRS = set(KNOWN_AGENTS.keys())

ALL_SCENARIOS     = ["simple", "medium", "complex"]
SCENARIO_LABELS   = {"simple": "Simples", "medium": "Médio", "complex": "Complexo"}

# Paleta de cores — uma por agente (até 12 agentes)
_BASE_PALETTE = [
    "#E64B35",  # vermelho
    "#4DBBD5",  # azul claro
    "#FF8C00",  # laranja
    "#036060",  # verde escuro
    "#3C5488",  # azul escuro
    "#6A0572",  # roxo escuro
    "#00A087",  # verde-água
    "#91D1C2",  # menta
    "#DC0000",  # vermelho vivo
    "#7E6148",  # marrom
    "#B09C85",  # bege
    "#8491B4",  # roxo claro
]

# Paleta alternativa por cenário (para modo --by-scenario)
SCENARIO_COLORS = {
    "simple":  "#4878CF",
    "medium":  "#6ACC65",
    "complex": "#D65F5F",
}

# Definição das três métricas
METRICS = {
    "rewards": {
        "npz_key":  "ep__rewards",
        "agg_key":  "rewards",
        "label":    "Recompensa Acumulada",
        "y_format": lambda x, _: format_br(x, precision=0),
        "unit":     "",
    },
    "delivery_time": {
        "npz_key":  "ep__delivery_time",
        "agg_key":  "delivery_time",
        "label":    "Tempo Efetivo Gasto",
        "y_format": lambda x, _: format_br(x, precision=0),
        "unit":     "(s)",
    },
    "distance": {
        "npz_key":  "ep__distance",
        "agg_key":  "distance",
        "label":    "Distância Percorrida",
        "y_format": lambda x, _: format_br(x, precision=0),
        "unit":     "(km)",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
#  Carregamento de dados
# ─────────────────────────────────────────────────────────────────────────────

def _load_npz(path: str) -> dict[str, list]:
    """Extrai séries por episódio de um arquivo NPZ."""
    result: dict[str, list] = {}
    with np.load(path, allow_pickle=False) as raw:
        for metric_key, meta in METRICS.items():
            npz_key = meta["npz_key"]
            if npz_key in raw.files:
                arr = raw[npz_key]
                result[metric_key] = arr.tolist()
    return result


def _load_json(path: str) -> dict[str, list]:
    """Extrai séries por episódio de um arquivo JSON no formato SimulationStats."""
    result: dict[str, list] = {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sim_list = data.get("sim", [])
    if sim_list:
        result["rewards"]       = [ep.get("reward")        for ep in sim_list]
        result["delivery_time"] = [ep.get("delivery_time") for ep in sim_list]
        result["distance"]      = [ep.get("distance")      for ep in sim_list]
    return result


def _load_aggregate_fallback(agent_dir: str) -> dict[str, list]:
    """
    Quando não há dados por episódio, reconstrói distribuição Normal
    a partir dos agregados (avg ± std_dev, N=30). Emite aviso.
    """
    agg: dict = {}

    npz_p  = os.path.join(agent_dir, "metrics_data.npz")
    json_p = os.path.join(agent_dir, "metrics_data.json")

    if os.path.exists(npz_p):
        try:
            with np.load(npz_p, allow_pickle=False) as raw:
                for key in raw.files:
                    parts = key.split("__")
                    if parts[0] == "agg" and len(parts) == 3:
                        _, metric, stat = parts
                        agg.setdefault(metric, {})[stat] = float(raw[key][0])
        except Exception:
            pass

    if not agg and os.path.exists(json_p):
        try:
            with open(json_p) as f:
                data = json.load(f)
            agg = data.get("aggregate", {})
        except Exception:
            pass

    if not agg:
        return {}

    warnings.warn(
        f"[AVISO] Dados por episódio não encontrados em '{agent_dir}'.\n"
        "  Usando média ± desvio padrão para aproximar a distribuição (N=30 simulado).\n"
        "  Os boxplots serão apenas ILUSTRATIVOS para esse agente.",
        UserWarning,
        stacklevel=4,
    )

    rng    = np.random.default_rng(seed=42)
    result = {}
    metric_map = {
        "rewards":       "rewards",
        "delivery_time": "delivery_time",
        "distance":      "distance",
    }
    for key, agg_name in metric_map.items():
        stats = agg.get(agg_name, {})
        if "avg" in stats and "std_dev" in stats:
            mu  = float(stats["avg"])
            std = max(float(stats["std_dev"]), 1e-9)
            result[key] = rng.normal(mu, std, 30).tolist()
    return result


def load_agent_data(agent_dir: str) -> dict[str, list]:
    """
    Carrega dados por episódio para um agente.
    Prioridade: NPZ > JSON > fallback (agregados → Normal simulada).
    Retorna dict {metric_key: [val_ep1, val_ep2, ...]}.
    """
    npz_p  = os.path.join(agent_dir, "metrics_data.npz")
    json_p = os.path.join(agent_dir, "metrics_data.json")

    if os.path.exists(npz_p):
        try:
            data = _load_npz(npz_p)
            if data:
                return data
        except Exception as e:
            warnings.warn(f"Erro ao ler NPZ em {agent_dir}: {e}")

    if os.path.exists(json_p):
        try:
            data = _load_json(json_p)
            if data:
                return data
        except Exception as e:
            warnings.warn(f"Erro ao ler JSON em {agent_dir}: {e}")

    return _load_aggregate_fallback(agent_dir)


def collect_data(
    results_dir: str,
    agents: list[str],
    scenarios: list[str],
    objective: int,
) -> dict:
    """
    Retorna:
      data[agent][scenario][metric_key] = [val_ep1, val_ep2, ...]
    """
    data: dict = {}
    for agent in agents:
        data[agent] = {}
        for scenario in scenarios:
            agent_dir = os.path.join(
                results_dir, f"obj_{objective}", f"{scenario}_scenario", agent
            )
            if not os.path.isdir(agent_dir):
                data[agent][scenario] = {}
                continue
            data[agent][scenario] = load_agent_data(agent_dir)
    return data


# ─────────────────────────────────────────────────────────────────────────────
#  Descoberta automática de agentes
# ─────────────────────────────────────────────────────────────────────────────

def discover_agents(results_dir: str, scenarios: list[str], objective: int) -> list[str]:
    """
    Varre os diretórios e retorna todos os agentes que possuem resultados,
    na ordem: heurísticas conhecidas → PPO (ordem alfabética).
    """
    found_heuristics: list[str] = []
    found_ppo: list[str]        = []

    for scenario in scenarios:
        base = os.path.join(results_dir, f"obj_{objective}", f"{scenario}_scenario")
        if not os.path.isdir(base):
            continue
        for entry in sorted(os.scandir(base), key=lambda e: e.name):
            if not entry.is_dir():
                continue
            name = entry.name
            has_data = (
                os.path.exists(os.path.join(entry.path, "metrics_data.npz")) or
                os.path.exists(os.path.join(entry.path, "metrics_data.json"))
            )
            if not has_data:
                continue
            if name in HEURISTIC_DIRS and name not in found_heuristics:
                found_heuristics.append(name)
            elif name not in HEURISTIC_DIRS and name not in found_ppo:
                found_ppo.append(name)

    # Garante a ordem canônica das heurísticas conhecidas
    canonical = list(KNOWN_AGENTS.keys())
    found_heuristics.sort(key=lambda x: canonical.index(x) if x in canonical else 99)

    return found_heuristics + found_ppo


def agent_label(agent: str) -> str:
    return KNOWN_AGENTS.get(agent, agent.replace("_", " ").title())


def build_color_map(agents: list[str]) -> dict[str, str]:
    return {a: _BASE_PALETTE[i % len(_BASE_PALETTE)] for i, a in enumerate(agents)}


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clean(values) -> list[float]:
    """Remove None e NaN de uma lista."""
    if values is None:
        return []
    return [float(v) for v in values if v is not None and np.isfinite(float(v))]


def _apply_rcparams(font_size: int, dpi: int) -> None:
    matplotlib.rcParams.update({
        "font.family":       "serif",
        "font.size":         font_size,
        "axes.labelsize":    font_size + 1,
        "axes.titlesize":    font_size + 2,
        "xtick.labelsize":   font_size - 1,
        "ytick.labelsize":   font_size - 1,
        "legend.fontsize":   font_size - 1,
        "figure.dpi":        dpi,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.30,
        "grid.linestyle":    "--",
        #"grid.axis":         "y",
    })


def _draw_boxplot_on_ax(
    ax: plt.Axes,
    data_matrix: list[list[float]],
    positions: np.ndarray,
    box_width: float,
    color: str,
    label: str,
    showfliers: bool,
    show_means: bool,
    show_mean_values: bool,
) -> None:
    """Desenha um conjunto de boxes para um único agente em um eixo."""
    bp = ax.boxplot(
        data_matrix,
        positions=positions,
        widths=box_width * 0.85,
        patch_artist=True,
        notch=False,
        showfliers=showfliers,
        showmeans=show_means,
        flierprops=dict(
            marker="o", markersize=3.5, alpha=0.55,
            markerfacecolor=color, markeredgecolor=color,
            linestyle="none",
        ),
        meanprops=dict(
            marker="D", markersize=4, markerfacecolor="white",
            markeredgecolor="black", markeredgewidth=0.8,
        ),
        medianprops=dict(color="black",   linewidth=1.8),
        whiskerprops=dict(color=color,    linewidth=1.2),
        capprops=dict(color=color,        linewidth=1.4),
        boxprops=dict(
            facecolor=color, alpha=0.70,
            linewidth=0.9, color=color,
        ),
    )
    # Guarda label no primeiro box para legenda posterior
    if bp["boxes"]:
        bp["boxes"][0].set_label(label)

    # Anota o valor numérico da média ao lado de cada losango
    if show_means and show_mean_values and bp["means"]:
        for mean_line in bp["means"]:
            x_val = mean_line.get_xdata()[0]
            y_val = mean_line.get_ydata()[0]
            if np.isfinite(y_val):
                ax.annotate(
                    format_br(y_val, precision=2),
                    xy=(x_val, y_val),
                    xytext=(4, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=6,
                    color="#222222",
                    bbox=dict(
                        boxstyle="round,pad=0.15",
                        facecolor="white",
                        edgecolor="lightgray",
                        alpha=0.85,
                        linewidth=0.5,
                    ),
                )


def _add_sample_annotation(
    ax: plt.Axes,
    data_matrix: list[list[float]],
    positions: np.ndarray,
    y_bottom: float,
) -> None:
    """Anota o N abaixo de cada box."""
    for pos, vals in zip(positions, data_matrix):
        n = len(vals)
        if n > 0:
            ax.text(
                pos, y_bottom, f"n={n}",
                ha="center", va="top", fontsize=6.5, color="#555555",
                transform=ax.get_xaxis_transform(),
            )

def format_br(x, precision=2):
    return f"{x:,.{precision}f}".replace(",", "_").replace(".", ",").replace("_", ".")

# ─────────────────────────────────────────────────────────────────────────────
#  Funções de plotagem principais
# ─────────────────────────────────────────────────────────────────────────────

def _plot_metric_ax(
    ax: plt.Axes,
    metric_key: str,
    data: dict,
    agents: list[str],
    scenarios: list[str],
    color_map: dict[str, str],
    by_scenario: bool,
    showfliers: bool,
    show_means: bool,
    show_mean_values: bool,
    annotate_n: bool,
    title: Optional[str],
) -> bool:
    """
    Plota um boxplot de uma métrica em `ax`.
    Retorna True se ao menos um box foi desenhado.

    Modos:
      by_scenario=False  → Eixo X = agentes, grupos = cenários
      by_scenario=True   → Eixo X = cenários, grupos = agentes
    """
    meta = METRICS[metric_key]

    if by_scenario:
        # ── Grupos = cenários, boxes = agentes ────────────────────────
        n_groups  = len(scenarios)
        n_boxes   = len(agents)
        x_labels  = [SCENARIO_LABELS.get(s, s) for s in scenarios]
        group_positions = np.arange(n_groups)
    else:
        # ── Grupos = agentes, boxes = cenários ────────────────────────
        n_groups  = len(agents)
        n_boxes   = len(scenarios)
        x_labels  = [agent_label(a) for a in agents]
        group_positions = np.arange(n_groups)

    group_width = 0.85
    box_width   = group_width / (n_boxes + 0.5)
    offsets     = np.linspace(
        -(n_boxes - 1) / 2 * box_width,
         (n_boxes - 1) / 2 * box_width,
        n_boxes,
    )

    has_data = False

    if by_scenario:
        for b_idx, agent in enumerate(agents):
            positions   = group_positions + offsets[b_idx]
            data_matrix = []
            for scenario in scenarios:
                vals = _clean(data.get(agent, {}).get(scenario, {}).get(metric_key))
                data_matrix.append(vals if vals else [np.nan])
            real = [d for d in data_matrix if not (len(d) == 1 and np.isnan(d[0]))]
            if not real:
                continue
            has_data = True
            _draw_boxplot_on_ax(
                ax, data_matrix, positions, box_width,
                color_map[agent], agent_label(agent),
                showfliers, show_means, show_mean_values,
            )
    else:
        for b_idx, scenario in enumerate(scenarios):
            positions   = group_positions + offsets[b_idx]
            data_matrix = []
            for agent in agents:
                vals = _clean(data.get(agent, {}).get(scenario, {}).get(metric_key))
                data_matrix.append(vals if vals else [np.nan])
            real = [d for d in data_matrix if not (len(d) == 1 and np.isnan(d[0]))]
            if not real:
                continue
            has_data = True
            color = SCENARIO_COLORS.get(scenario, _BASE_PALETTE[b_idx % len(_BASE_PALETTE)])
            _draw_boxplot_on_ax(
                ax, data_matrix, positions, box_width,
                color, SCENARIO_LABELS.get(scenario, scenario),
                showfliers, show_means, show_mean_values,
            )

    # Formatação do eixo
    ax.set_xticks(group_positions)
    ax.set_xticklabels(x_labels, rotation=15 if not by_scenario else 0, ha="right" if not by_scenario else "center")
    ylabel = meta["label"]
    if meta["unit"]:
        ylabel += f" {meta['unit']}"
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(FuncFormatter(meta["y_format"]))

    if title:
        ax.set_title(title, pad=8)
    else:
        ax.set_title(meta["label"], pad=8)

    if not has_data:
        ax.text(0.5, 0.5, "Sem dados", transform=ax.transAxes,
                ha="center", va="center", color="gray", fontsize=11)

    return has_data


def _plot_single_scenario_ax(
    ax: plt.Axes,
    metric_key: str,
    data: dict,
    agents: list[str],
    scenario: str,
    color_map: dict[str, str],
    showfliers: bool,
    show_means: bool,
    show_mean_values: bool,
    annotate_n: bool,
    sharey_ax: Optional[plt.Axes] = None,
) -> bool:
    """
    Plota um boxplot por agente para um único cenário.
    Usado por plot_per_scenario (--split-scenarios).
    Retorna True se ao menos um box foi desenhado.
    """
    meta      = METRICS[metric_key]
    positions = np.arange(len(agents))
    box_width = 0.55
    has_data  = False

    for i, agent in enumerate(agents):
        vals = _clean(data.get(agent, {}).get(scenario, {}).get(metric_key))
        if not vals:
            continue
        has_data = True
        _draw_boxplot_on_ax(
            ax,
            data_matrix=[vals],
            positions=np.array([positions[i]]),
            box_width=box_width,
            color=color_map[agent],
            label=agent_label(agent),
            showfliers=showfliers,
            show_means=show_means,
            show_mean_values=show_mean_values,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [agent_label(a) for a in agents],
        rotation=20, ha="right",
    )
    ax.set_title(SCENARIO_LABELS.get(scenario, scenario), pad=8)

    ylabel = meta["label"]
    if meta["unit"]:
        ylabel += f" {meta['unit']}"
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(FuncFormatter(meta["y_format"]))

    if not has_data:
        ax.text(0.5, 0.5, "Sem dados", transform=ax.transAxes,
                ha="center", va="center", color="gray", fontsize=11)

    return has_data


def _add_legend(
    fig: plt.Figure,
    agents: list[str],
    scenarios: list[str],
    color_map: dict[str, str],
    by_scenario: bool,
    n_cols: Optional[int],
    legend_stats: bool = False,
    show_means: bool = False,
) -> None:
    """Adiciona legenda única abaixo da figura.

    Se legend_stats=True, acrescenta entradas para Mediana (linha preta)
    e, se show_means=True, para Média (losango branco com borda preta).
    """
    from matplotlib.lines import Line2D

    if by_scenario:
        handles = [
            mpatches.Patch(
                facecolor=color_map[a], alpha=0.75,
                label=agent_label(a), edgecolor="black", linewidth=0.5,
            )
            for a in agents
        ]
        ncol_base = n_cols or len(agents)
    else:
        handles = [
            mpatches.Patch(
                facecolor=SCENARIO_COLORS.get(s, _BASE_PALETTE[i]),
                alpha=0.75,
                label=SCENARIO_LABELS.get(s, s),
                edgecolor="black", linewidth=0.5,
            )
            for i, s in enumerate(scenarios)
        ]
        ncol_base = n_cols or len(scenarios)

    # ── Entradas de estatísticas (mediana / média) ─────────────────────────
    stat_handles: list = []
    if legend_stats:
        stat_handles.append(
            Line2D(
                [0], [0],
                color="black", linewidth=1.8,
                label="Mediana",
            )
        )
        if show_means:
            stat_handles.append(
                Line2D(
                    [0], [0],
                    marker="D", markersize=5,
                    markerfacecolor="white", markeredgecolor="black",
                    markeredgewidth=0.8,
                    linestyle="none",
                    label="Média",
                )
            )

    all_handles = handles + stat_handles
    # Se há entradas de stats, adiciona 1 coluna extra para elas ficarem à parte
    ncol = (n_cols or ncol_base) + (1 if stat_handles else 0)

    fig.legend(
        handles=all_handles,
        loc="lower center",
        ncol=ncol,
        frameon=True,
        framealpha=0.92,
        edgecolor="lightgray",
        bbox_to_anchor=(0.5, -0.04),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Geração de figuras
# ─────────────────────────────────────────────────────────────────────────────

def plot_combined(
    data: dict,
    agents: list[str],
    scenarios: list[str],
    metrics: list[str],
    output_path: str,
    by_scenario: bool,
    showfliers: bool,
    show_means: bool,
    show_mean_values: bool,
    annotate_n: bool,
    figsize: tuple[float, float],
    dpi: int,
    font_size: int,
    suptitle: Optional[str],
    legend_cols: Optional[int],
    legend_stats: bool = False,
) -> None:
    """Cria uma figura com N subplots (um por métrica) lado a lado."""
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, metric_key in zip(axes, metrics):
        _plot_metric_ax(
            ax, metric_key, data, agents, scenarios,
            build_color_map(agents), by_scenario,
            showfliers, show_means, show_mean_values, annotate_n, None,
        )

    if suptitle:
        fig.suptitle(suptitle, fontsize=font_size + 3, y=1.01)

    _add_legend(fig, agents, scenarios, build_color_map(agents), by_scenario,
                legend_cols, legend_stats=legend_stats, show_means=show_means)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
    print(f"  ✔ Figura salva: {output_path}")
    plt.close(fig)


def plot_split(
    data: dict,
    agents: list[str],
    scenarios: list[str],
    metrics: list[str],
    output_dir: str,
    prefix: str,
    fmt: str,
    by_scenario: bool,
    showfliers: bool,
    show_means: bool,
    show_mean_values: bool,
    annotate_n: bool,
    figsize: tuple[float, float],
    dpi: int,
    font_size: int,
    suptitle: Optional[str],
    legend_cols: Optional[int],
    legend_stats: bool = False,
) -> None:
    """Salva cada métrica em um arquivo separado."""
    for metric_key in metrics:
        meta   = METRICS[metric_key]
        fname  = f"{prefix}_{metric_key}.{fmt}"
        output = os.path.join(output_dir, fname)

        fig, ax = plt.subplots(1, 1, figsize=(figsize[0] / max(len(metrics), 1), figsize[1]))

        _plot_metric_ax(
            ax, metric_key, data, agents, scenarios,
            build_color_map(agents), by_scenario,
            showfliers, show_means, show_mean_values, annotate_n,
            title=suptitle,
        )

        _add_legend(fig, agents, scenarios, build_color_map(agents), by_scenario,
                    legend_cols, legend_stats=legend_stats, show_means=show_means)
        fig.tight_layout(rect=[0, 0.08, 1, 1])
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(output, bbox_inches="tight", dpi=dpi)
        print(f"  ✔ Figura salva: {output}")
        plt.close(fig)


def plot_per_scenario(
    data: dict,
    agents: list[str],
    scenarios: list[str],
    metrics: list[str],
    output_dir: str,
    prefix: str,
    fmt: str,
    showfliers: bool,
    show_means: bool,
    show_mean_values: bool,
    annotate_n: bool,
    figsize: tuple[float, float],
    dpi: int,
    font_size: int,
    suptitle: Optional[str],
    legend_cols: Optional[int],
    legend_stats: bool = False,
) -> None:
    """
    Modo --split-scenarios (requer --by-scenario).
    Para cada métrica gera 1 arquivo com N subplots lado a lado,
    um por cenário. Eixo X = agentes, cor = agente.
    Cada subplot possui escala Y independente.
    Legenda unificada no rodapé.

    Nomes dos arquivos:
      {prefix}_{metric_key}_scenarios.{fmt}
    """
    color_map = build_color_map(agents)
    n_sc      = len(scenarios)

    for metric_key in metrics:
        meta  = METRICS[metric_key]
        fname = f"{prefix}_{metric_key}_scenarios.{fmt}"
        out   = os.path.join(output_dir, fname)

        # Largura proporcional ao nº de cenários; altura fixa
        # sharey removido: cada subplot tem sua própria escala Y
        fw = figsize[0] * (n_sc / 3)
        fh = figsize[1]
        fig, axes = plt.subplots(1, n_sc, figsize=(fw, fh))
        if n_sc == 1:
            axes = [axes]

        # Título geral da figura = métrica (+ suptitle opcional)
        base_title = suptitle or meta["label"]
        if meta["unit"]:
            base_title = f"{base_title} {meta['unit']}" if suptitle else f"{meta['label']} {meta['unit']}"
        fig.suptitle(base_title, fontsize=font_size + 3, y=1.01)

        for col, (ax, scenario) in enumerate(zip(axes, scenarios)):
            # Cada subplot tem escala independente (sharey_ax=None para todos)
            _plot_single_scenario_ax(
                ax, metric_key, data, agents, scenario,
                color_map, showfliers, show_means, show_mean_values,
                annotate_n, sharey_ax=None,
            )

        _add_legend(
            fig, agents, scenarios, color_map,
            by_scenario=True,       # por agente → usa color_map de agentes
            n_cols=legend_cols,
            legend_stats=legend_stats,
            show_means=show_means,
        )
        fig.tight_layout(rect=[0, 0.10, 1, 1])
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(out, bbox_inches="tight", dpi=dpi)
        print(f"  ✔ Figura salva: {out}")
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Gera boxplots de Recompensa, Tempo Efetivo e Distância Percorrida\n"
            "para heurísticas e modelos PPO avaliados no food-delivery-gym."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Exemplos:
  # Só heurísticas principais, cenário médio
  python generate_boxplots.py --agents random nearest_driver lowest_marginal_route_cost \\
         --scenarios medium

  # Comparar dois modelos PPO com as heurísticas, objetivo 5
  python generate_boxplots.py --agents random nearest_driver ppo_18M ppo_36M --objective 5

  # Agentes no eixo X, cenários como grupos de boxes (padrão inverso)
  python generate_boxplots.py --by-scenario

  # Salvar cada métrica em arquivo separado, alta resolução
  python generate_boxplots.py --split --dpi 600 --fmt pdf

  # Ocultar outliers, mostrar médias com valor numérico, anotar N amostras
  python generate_boxplots.py --no-fliers --show-means --show-mean-values --annotate-n

  # Um PNG por métrica com 3 subplots (simples/médio/complexo), legenda unificada
  python generate_boxplots.py --by-scenario --split-scenarios

  # Idem, com entradas de mediana e média na legenda
  python generate_boxplots.py --by-scenario --split-scenarios --show-means --legend-stats

  # Selecionar apenas recompensa e distância
  python generate_boxplots.py --metrics rewards distance
""",
    )

    # ── Dados ──────────────────────────────────────────────────────────────
    data = parser.add_argument_group("Dados")
    data.add_argument(
        "--results-dir", "-r",
        default=DEFAULT_RESULTS_DIR,
        metavar="DIR",
        help=f"Diretório raiz com os resultados.\nPadrão: {DEFAULT_RESULTS_DIR}",
    )
    data.add_argument(
        "--objective", "-o",
        type=int,
        default=DEFAULT_OBJECTIVE,
        metavar="N",
        help=f"Número do objetivo (obj_N/). Padrão: {DEFAULT_OBJECTIVE}",
    )
    data.add_argument(
        "--scenarios", "-s",
        nargs="+",
        default=ALL_SCENARIOS,
        choices=ALL_SCENARIOS,
        metavar="SCENARIO",
        help=(
            f"Cenários a incluir (espaço-separados).\n"
            f"Opções: {ALL_SCENARIOS}\n"
            f"Padrão: todos ({ALL_SCENARIOS})"
        ),
    )
    data.add_argument(
        "--agents", "-a",
        nargs="+",
        default=None,
        metavar="AGENT",
        help=(
            "Nomes dos diretórios dos agentes a incluir (na ordem desejada).\n"
            "Se omitido, todos os agentes com dados são descobertos automaticamente.\n"
            "Exemplos de nomes: random  nearest_driver  ppo_18M_steps"
        ),
    )
    data.add_argument(
        "--exclude-agents",
        nargs="+",
        default=[],
        metavar="AGENT",
        help="Agentes a excluir (mesmo se descobertos automaticamente).",
    )

    # ── Métricas ────────────────────────────────────────────────────────────
    met = parser.add_argument_group("Métricas")
    met.add_argument(
        "--metrics", "-M",
        nargs="+",
        default=list(METRICS.keys()),
        choices=list(METRICS.keys()),
        metavar="METRIC",
        help=(
            "Métricas a plotar (espaço-separadas).\n"
            f"Opções: {list(METRICS.keys())}\n"
            "Padrão: todas"
        ),
    )

    # ── Layout ──────────────────────────────────────────────────────────────
    layout = parser.add_argument_group("Layout e visualização")
    layout.add_argument(
        "--by-scenario",
        action="store_true",
        help=(
            "Agrupa boxes por agente no eixo X e usa cenários como grupos de boxes.\n"
            "Padrão: agrupa por cenário no eixo X, agentes como grupos de boxes."
        ),
    )
    layout.add_argument(
        "--split-scenarios",
        action="store_true",
        help=(
            "Gera 1 PNG por métrica contendo N subplots (um por cenário).\n"
            "Cada subplot mostra os agentes no eixo X com suas respectivas boxes.\n"
            "Cada subplot tem escala Y independente.\n"
            "Requer --by-scenario. A legenda é unificada para todos os subplots."
        ),
    )
    layout.add_argument(
        "--split",
        action="store_true",
        help="Salva cada métrica em um arquivo separado em vez de uma figura única.",
    )
    layout.add_argument(
        "--no-fliers",
        action="store_true",
        help="Oculta os outliers (pontos além dos whiskers).",
    )
    layout.add_argument(
        "--show-means",
        action="store_true",
        help="Exibe a média como um losango dentro de cada box.",
    )
    layout.add_argument(
        "--show-mean-values",
        action="store_true",
        help=(
            "Anota o valor numérico da média ao lado de cada losango.\n"
            "Requer --show-means para ter efeito."
        ),
    )
    layout.add_argument(
        "--annotate-n",
        action="store_true",
        help="Anota o número de amostras (N) abaixo de cada box.",
    )
    layout.add_argument(
        "--suptitle",
        default=None,
        metavar="TEXTO",
        help="Título geral da figura.",
    )
    layout.add_argument(
        "--legend-cols",
        type=int,
        default=None,
        metavar="N",
        help="Número de colunas da legenda. Padrão: automático.",
    )
    layout.add_argument(
        "--legend-stats",
        action="store_true",
        help=(
            "Adiciona entradas na legenda para Mediana (linha preta) e,\n"
            "se --show-means estiver ativo, para Média (losango branco)."
        ),
    )

    # ── Saída ───────────────────────────────────────────────────────────────
    out = parser.add_argument_group("Saída")
    out.add_argument(
        "--output-dir", "-od",
        default=DEFAULT_OUTPUT_DIR,
        metavar="DIR",
        help=f"Diretório de saída. Padrão: {DEFAULT_OUTPUT_DIR}",
    )
    out.add_argument(
        "--prefix",
        default="boxplot",
        metavar="PREFIXO",
        help="Prefixo do nome do arquivo de saída. Padrão: boxplot",
    )
    out.add_argument(
        "--fmt",
        choices=["pdf", "png", "svg"],
        default="pdf",
        help="Formato de saída. Padrão: pdf",
    )
    out.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=[16, 5],
        metavar=("L", "A"),
        help="Dimensões da figura (largura altura) em polegadas. Padrão: 16 5",
    )
    out.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolução da figura em DPI. Padrão: 300",
    )
    out.add_argument(
        "--font-size",
        type=int,
        default=9,
        metavar="N",
        help="Tamanho base da fonte. Padrão: 9",
    )

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  Ponto de entrada
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    _apply_rcparams(font_size=args.font_size, dpi=args.dpi)

    # ── Resolver lista de agentes ──────────────────────────────────────────
    if args.agents:
        agents = args.agents
    else:
        agents = discover_agents(args.results_dir, args.scenarios, args.objective)
        if agents:
            print(f"Agentes descobertos automaticamente: {agents}")
        else:
            print(
                "[AVISO] Nenhum agente encontrado. "
                "Verifique --results-dir e --objective."
            )
            return

    # Remove excluídos
    agents = [a for a in agents if a not in args.exclude_agents]

    if not agents:
        print("[AVISO] Nenhum agente restante após exclusões.")
        return

    print(f"Agentes    : {agents}")
    print(f"Cenários   : {args.scenarios}")
    print(f"Objetivo   : obj_{args.objective}")
    print(f"Métricas   : {args.metrics}")
    print(f"Modo eixo  : {'agentes no eixo X' if args.by_scenario else 'cenários no eixo X'}")

    # ── Carregar dados ─────────────────────────────────────────────────────
    data = collect_data(
        args.results_dir, agents, args.scenarios, args.objective
    )

    # ── Gerar figuras ──────────────────────────────────────────────────────
    kwargs = dict(
        data=data,
        agents=agents,
        scenarios=args.scenarios,
        metrics=args.metrics,
        by_scenario=args.by_scenario,
        showfliers=not args.no_fliers,
        show_means=args.show_means,
        show_mean_values=args.show_mean_values,
        annotate_n=args.annotate_n,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
        font_size=args.font_size,
        suptitle=args.suptitle,
        legend_cols=args.legend_cols,
        legend_stats=args.legend_stats,
    )

    if args.split_scenarios:
        if not args.by_scenario:
            print("[AVISO] --split-scenarios requer --by-scenario. Adicionando automaticamente.")
        plot_per_scenario(
            **{k: v for k, v in kwargs.items() if k not in ("by_scenario",)},
            output_dir=args.output_dir,
            prefix=args.prefix,
            fmt=args.fmt,
        )
    elif args.split:
        plot_split(
            **kwargs,
            output_dir=args.output_dir,
            prefix=args.prefix,
            fmt=args.fmt,
        )
    else:
        output_path = os.path.join(
            args.output_dir, f"{args.prefix}_obj{args.objective}.{args.fmt}"
        )
        plot_combined(**kwargs, output_path=output_path)

if __name__ == "__main__":
    main()