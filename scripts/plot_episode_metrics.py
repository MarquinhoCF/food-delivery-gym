"""
Plota por episódio, gerando UMA FIGURA POR SÉRIE de experimento PPO:
  • Retorno (total_rewards)
  • Tamanho do episódio em passos (episode_lengths)
  • Último tempo SimPy do episódio (simpy_last_time_steps)

Cada figura é salva como:
    episode_metrics_<label_sanitizado>.png

Uso:
    python -m scripts.plot_episode_metrics
    python -m scripts.plot_episode_metrics --base-dir results/obj_11/medium/
    python -m scripts.plot_episode_metrics. \
        --dirs ppo_otimizado_trained_18M_steps \
               ppo_otimizado_trained_18M_steps_with_penalty \
               ppo_otimizado_trained_18M_steps_with_bonus \
        --labels "Base" "Penalty" "Bonus"
"""

import argparse
import os
import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sns.set_theme(style="darkgrid", font_scale=1.05)

COLORS  = ["#3A86FF", "#FF6B6B", "#06D6A0"]
MARKERS = ["o",        "s",       "D"      ]

DEFAULT_DIRS = [
    "ppo_otimizado_trained_18M_steps",
    "ppo_otimizado_trained_18M_steps_with_penalty",
    "ppo_otimizado_trained_18M_steps_with_bonus",
]
DEFAULT_LABELS = ["Base", "With Penalty", "With Bonus"]


# ── Carregamento ───────────────────────────────────────────────────────────

def load_series(directory: str, label: str) -> dict | None:
    npz_path = os.path.join(directory, "metrics_data.npz")
    if not os.path.exists(npz_path):
        print(f"⚠  Não encontrado: {npz_path}")
        return None
    try:
        raw = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        print(f"✗ Erro ao abrir {npz_path}: {e}")
        return None

    def extract(key):
        if key not in raw:
            return None
        arr = raw[key]
        if arr.ndim == 0 or arr.size == 0:
            return None
        return arr.astype(float)

    rewards = extract("total_rewards")
    lengths = extract("episode_lengths")
    simpy_t = extract("simpy_last_time_steps")

    if rewards is None:
        print(f"✗ 'total_rewards' ausente em {npz_path}")
        return None

    n = len(rewards)
    print(
        f"✓ {label}: {n} episódios | "
        f"episode_lengths={'ok' if lengths is not None else 'N/A'} | "
        f"simpy_last_time_steps={'ok' if simpy_t is not None else 'N/A'}"
    )
    return {"label": label, "dir": directory, "n": n,
            "rewards": rewards, "lengths": lengths, "simpy_t": simpy_t}


# ── Helpers de plot ────────────────────────────────────────────────────────

def _x(n):
    return np.arange(1, n + 1)


def _unavailable(ax, key):
    ax.text(0.5, 0.5,
            f"'{key}' não disponível\n(ausente no metrics_data.npz)",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=9, color="gray", style="italic")


def _plot_line(ax, s, key, ylabel, title, color, marker):
    arr = s.get(key)
    if arr is None:
        _unavailable(ax, key)
    else:
        mean, std = np.nanmean(arr), np.nanstd(arr)
        ax.plot(_x(len(arr)), arr, color=color, alpha=0.55,
                linewidth=1.2, marker=marker, markersize=5)
        ax.axhline(mean, color=color, linewidth=2.0, linestyle=":",
                   label=f"μ = {mean:.2f}   σ = {std:.2f}")
        ax.legend(fontsize=9)
    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.set_xlabel("Episódio", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)


def _plot_box(ax, s, key, ylabel, title, color):
    arr = s.get(key)
    if arr is None:
        _unavailable(ax, key)
        ax.set_title(title, fontweight="bold", fontsize=11)
        return
    valid = arr[~np.isnan(arr)]
    bp = ax.boxplot([valid], labels=[s["label"]], patch_artist=True,
                    medianprops=dict(color="white", linewidth=2.2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    flierprops=dict(marker="x", markersize=5, alpha=0.5))
    bp["boxes"][0].set_facecolor(color)
    bp["boxes"][0].set_alpha(0.70)
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.15, 0.15, size=len(valid))
    ax.scatter(np.ones(len(valid)) + jitter, valid,
               color=color, alpha=0.55, s=30, zorder=3)
    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=9)


# ── Figura individual ──────────────────────────────────────────────────────

def build_figure_for_series(s, output, color, marker):
    fig = plt.figure(figsize=(13, 16))
    fig.suptitle(
        f"Análise por Episódio  ·  {s['label']}  ({s['n']} simulações)",
        fontsize=13, fontweight="bold", y=0.995,
    )
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.52, wspace=0.35)

    ax_ret   = fig.add_subplot(gs[0, :])
    ax_len   = fig.add_subplot(gs[1, :])
    ax_simpy = fig.add_subplot(gs[2, :])
    ax_b_ret = fig.add_subplot(gs[3, 0])
    ax_b_len = fig.add_subplot(gs[3, 1])
    ax_b_smp = fig.add_subplot(gs[3, 2])

    _plot_line(ax_ret,   s, "rewards", "Retorno",
               "Retorno por Episódio",           color, marker)
    _plot_line(ax_len,   s, "lengths", "Passos",
               "Tamanho do Episódio (passos)",   color, marker)
    _plot_line(ax_simpy, s, "simpy_t", "Tempo SimPy (u.t.)",
               "Último Tempo SimPy do Episódio", color, marker)

    _plot_box(ax_b_ret, s, "rewards", "Retorno",
              "Distribuição do Retorno",     color)
    _plot_box(ax_b_len, s, "lengths", "Passos",
              "Distribuição dos Passos",     color)
    _plot_box(ax_b_smp, s, "simpy_t", "Tempo SimPy (u.t.)",
              "Distribuição do Tempo SimPy", color)

    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"  ✅ Salvo: {output}")
    plt.show()
    plt.close(fig)


# ── Sumário no terminal ────────────────────────────────────────────────────

def print_summary(series_list):
    SEP = "=" * 72
    FMT = "{:<24} {:>4}  {:>9} {:>9}  {:>8} {:>8}  {:>8} {:>8}"
    print(f"\n{SEP}")
    print(FMT.format("SÉRIE", "N",
                     "RET μ", "RET σ", "LEN μ", "LEN σ", "SIMPY μ", "SIMPY σ"))
    print(SEP)
    for s in series_list:
        r  = s["rewards"]
        l  = s["lengths"]  if s["lengths"]  is not None else np.array([np.nan])
        st = s["simpy_t"]  if s["simpy_t"]  is not None else np.array([np.nan])
        print(FMT.format(
            s["label"][:24], s["n"],
            f"{np.nanmean(r):>9.2f}", f"{np.nanstd(r):>9.2f}",
            f"{np.nanmean(l):>8.1f}", f"{np.nanstd(l):>8.1f}",
            f"{np.nanmean(st):>8.1f}", f"{np.nanstd(st):>8.1f}",
        ))
    print(SEP)


def _sanitize(label):
    return re.sub(r"[^\w\-]", "_", label).strip("_").lower()


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Gera UMA FIGURA POR SÉRIE com retorno, episode_length e simpy_time_step."
    )
    parser.add_argument("--base-dir", "-b", type=str, default=None)
    parser.add_argument("--dirs", "-d", nargs="+", default=None)
    parser.add_argument("--labels", "-l", nargs="+", default=DEFAULT_LABELS)
    parser.add_argument("--output-dir", "-o", type=str, default=".",
                        help="Pasta onde salvar os PNGs (default: pasta atual)")
    args = parser.parse_args()

    if args.dirs:
        dirs = args.dirs
    elif args.base_dir:
        dirs = [os.path.join(args.base_dir, d) for d in DEFAULT_DIRS]
    else:
        dirs = DEFAULT_DIRS

    labels = list(args.labels)
    if len(labels) < len(dirs):
        labels += [os.path.basename(d) for d in dirs[len(labels):]]

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n📂 Carregando {len(dirs)} séries...")
    series_list = []
    for d, lbl in zip(dirs, labels):
        s = load_series(d, lbl)
        if s is not None:
            series_list.append(s)

    if not series_list:
        print("❌ Nenhuma série válida. Verifique os caminhos.")
        return

    print_summary(series_list)
    print(f"\n📊 Gerando {len(series_list)} figura(s)...")

    for s, color, marker in zip(series_list, COLORS, MARKERS):
        fname  = f"episode_metrics_{_sanitize(s['label'])}.png"
        output = os.path.join(args.output_dir, fname)
        build_figure_for_series(s, output, color, marker)

    print("\n✅ Concluído.")


if __name__ == "__main__":
    main()