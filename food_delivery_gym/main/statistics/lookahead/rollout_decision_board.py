from __future__ import annotations

import json
import os
from typing import Any

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from food_delivery_gym.main.statistics.boards.board import Board


class RolloutDecisionBoard(Board):
    """
    Visualização pós-episódio das decisões do RolloutOptimizerGym.

    Para cada decisão gera uma figura 1x2:
      - esquerda: barras immediate_reward vs alpha*rollout_value (Q = soma)
      - direita: árvore de decisão com trajetórias internas do rollout
        (ação candidata → passos da política base ao longo do horizonte)
    """

    # Layout da árvore (coordenadas em eixos abstratos)
    _NODE_W = 1.35
    _NODE_H = 0.72
    _X_GAP = 1.85
    _Y_GAP = 1.15

    def __init__(self, decision_log: list[dict]) -> None:
        super().__init__(metrics=[])
        self.decision_log = decision_log

    def view(self) -> None:
        for decision in self.decision_log:
            fig = self._build_figure(decision)
            plt.show()
            plt.close(fig)

    def save(self, dir_path: str) -> None:
        matplotlib.use("Agg")
        out_dir = os.path.join(dir_path, "rollout_decisions")
        os.makedirs(out_dir, exist_ok=True)

        # Uma figura por vez: evita acumular centenas de Figure abertas.
        for decision in self.decision_log:
            fig = self._build_figure(decision)
            idx = int(decision.get("decision_idx", 0)) + 1
            name = f"decision_{idx:03d}.png"
            fig.savefig(os.path.join(out_dir, name), dpi=150, bbox_inches="tight")
            plt.close(fig)

    def dump_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.decision_log, f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: str) -> "RolloutDecisionBoard":
        with open(path, "r", encoding="utf-8") as f:
            decision_log = json.load(f)
        return cls(decision_log)

    def _build_figure(self, decision: dict[str, Any]) -> Figure:
        candidates = decision.get("candidates") or []
        chosen_action = decision.get("chosen_action")
        alpha = float(decision.get("alpha", 1.0))
        order_id = decision.get("order_id")
        sim_time = decision.get("sim_time")
        horizon = decision.get("horizon")

        max_traj = max((len(c.get("trajectory") or []) for c in candidates), default=0)
        n_cand = max(len(candidates), 1)
        tree_w = 2 + max_traj
        fig_w = max(14.0, 4.5 + tree_w * 1.6)
        fig_h = max(6.0, 2.5 + n_cand * 1.1)

        fig, (ax_bars, ax_tree) = plt.subplots(
            1, 2, figsize=(fig_w, fig_h), gridspec_kw={"width_ratios": [1.0, 2.2]}
        )
        fig.suptitle(
            f"Decision {int(decision.get('decision_idx', 0)) + 1}  |  "
            f"order_id={order_id}  |  sim_time={sim_time}  |  "
            f"H={horizon if horizon is not None else 'inf'}",
            fontsize=13,
            fontweight="bold",
        )

        self._draw_q_bars(ax_bars, candidates, chosen_action, alpha)
        self._draw_decision_tree(ax_tree, decision, candidates, chosen_action, alpha)

        fig.tight_layout()
        return fig

    def _draw_q_bars(
        self,
        ax,
        candidates: list[dict],
        chosen_action: int | None,
        alpha: float,
    ) -> None:
        if not candidates:
            ax.set_title("No candidates")
            ax.axis("off")
            return

        actions = [c["action"] for c in candidates]
        immediate = [c["immediate_reward"] for c in candidates]
        discounted_rollout = [alpha * c["rollout_value"] for c in candidates]
        q_values = [c["q_value"] for c in candidates]
        labels = [f"a={c['action']}\nid={c['driver_id']}" for c in candidates]

        x = range(len(actions))
        width = 0.55

        bars_imm = ax.bar(x, immediate, width, label="immediate_reward", color="#4C78A8")
        bars_roll = ax.bar(
            x,
            discounted_rollout,
            width,
            bottom=immediate,
            label=f"α·rollout_value (α={alpha:g})",
            color="#F58518",
        )

        for i, (imm_bar, roll_bar, q) in enumerate(zip(bars_imm, bars_roll, q_values)):
            if actions[i] == chosen_action:
                imm_bar.set_edgecolor("black")
                imm_bar.set_linewidth(2.5)
                imm_bar.set_hatch("//")
                roll_bar.set_edgecolor("black")
                roll_bar.set_linewidth(2.5)
                roll_bar.set_hatch("//")
            top = imm_bar.get_height() + roll_bar.get_height()
            ax.text(
                imm_bar.get_x() + imm_bar.get_width() / 2,
                top,
                f"{q:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Q components")
        ax.set_title("Q-values per driver (stacked)")
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.legend(loc="best", fontsize=8)
        ax.grid(axis="y", linestyle=":", alpha=0.5)

    def _draw_decision_tree(
        self,
        ax,
        decision: dict,
        candidates: list[dict],
        chosen_action: int | None,
        alpha: float,
    ) -> None:
        ax.set_title("Rollout decision tree (candidate → base-policy trajectory)")
        ax.set_aspect("equal")
        ax.axis("off")

        if not candidates:
            ax.text(0.5, 0.5, "No candidates", ha="center", va="center", transform=ax.transAxes)
            return

        n = len(candidates)
        # y das branches: ordenado por driver_id (menor id no topo)
        ordered = sorted(range(n), key=lambda i: candidates[i]["driver_id"])
        y_of = {i: (n - 1 - rank) * self._Y_GAP for rank, i in enumerate(ordered)}

        root_xy = (0.0, (n - 1) * self._Y_GAP / 2.0)
        root_label = (
            f"order {decision.get('order_id')}\n"
            f"t={decision.get('sim_time')}\n"
            f"choose driver"
        )
        self._draw_node(ax, root_xy, root_label, face="#E8E8E8", edge="#333333", lw=1.5)

        positions: dict[tuple[int, int], tuple[float, float]] = {}
        # depth 0 = candidate action node
        for i, cand in enumerate(candidates):
            x = self._X_GAP
            y = y_of[i]
            positions[(i, 0)] = (x, y)

            is_chosen = cand["action"] == chosen_action
            face = "#C7E9C0" if is_chosen else "#DEEBF7"
            edge = "#006D2C" if is_chosen else "#3182BD"
            lw = 2.4 if is_chosen else 1.2
            chosen_tag = " ★ CHOSEN" if is_chosen else ""
            label = (
                f"a={cand['action']}  drv={cand['driver_id']}{chosen_tag}\n"
                f"r₀={cand['immediate_reward']:.2f}\n"
                f"Q={cand['q_value']:.2f}"
            )
            self._draw_node(ax, (x, y), label, face=face, edge=edge, lw=lw)
            self._draw_edge(ax, root_xy, (x, y), label=f"try a={cand['action']}")

            traj = cand.get("trajectory") or []
            prev = (x, y)
            for step in traj:
                depth = int(step["step"]) + 1
                nx = (depth + 1) * self._X_GAP
                ny = y
                positions[(i, depth)] = (nx, ny)
                term = step.get("terminated") or step.get("truncated")
                face_s = "#FEE0D2" if term else "#FFF5EB"
                edge_s = "#A63603" if term else "#D94801"
                label_s = (
                    f"base a={step['action']} drv={step.get('driver_id')}\n"
                    f"ord={step.get('order_id')}  r={step['reward']:.2f}\n"
                    f"αᵏr={step['discounted_reward']:.2f}"
                )
                self._draw_node(ax, (nx, ny), label_s, face=face_s, edge=edge_s, lw=1.1)
                edge_lbl = f"k={step['step']}"
                self._draw_edge(ax, prev, (nx, ny), label=edge_lbl)
                prev = (nx, ny)

            if not traj:
                # folha vazia: episódio já terminou após a ação candidata
                nx = 2 * self._X_GAP
                self._draw_node(
                    ax,
                    (nx, y),
                    "end\n(no rollout)",
                    face="#F0F0F0",
                    edge="#888888",
                    lw=1.0,
                )
                self._draw_edge(ax, (x, y), (nx, y), label="")

        # bounding box
        all_x = [root_xy[0]] + [p[0] for p in positions.values()]
        all_y = [root_xy[1]] + [p[1] for p in positions.values()]
        # include empty-traj leafs
        if any(not (c.get("trajectory") or []) for c in candidates):
            all_x.append(2 * self._X_GAP)

        pad_x = self._NODE_W
        pad_y = self._NODE_H
        ax.set_xlim(min(all_x) - pad_x, max(all_x) + pad_x)
        ax.set_ylim(min(all_y) - pad_y, max(all_y) + pad_y)

        ax.text(
            0.01,
            0.01,
            "Green = chosen candidate  |  Orange = base-policy rollout steps",
            transform=ax.transAxes,
            fontsize=8,
            color="#555555",
            va="bottom",
        )

    def _draw_node(
        self,
        ax,
        xy: tuple[float, float],
        text: str,
        face: str,
        edge: str,
        lw: float,
    ) -> None:
        x, y = xy
        box = FancyBboxPatch(
            (x - self._NODE_W / 2, y - self._NODE_H / 2),
            self._NODE_W,
            self._NODE_H,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor=face,
            edgecolor=edge,
            linewidth=lw,
            zorder=3,
        )
        ax.add_patch(box)
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=7,
            zorder=4,
            family="monospace",
        )

    def _draw_edge(
        self,
        ax,
        start: tuple[float, float],
        end: tuple[float, float],
        label: str = "",
    ) -> None:
        # conecta borda direita do nó de origem à borda esquerda do destino
        x0 = start[0] + self._NODE_W / 2
        y0 = start[1]
        x1 = end[0] - self._NODE_W / 2
        y1 = end[1]
        arrow = FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.0,
            color="#666666",
            zorder=2,
        )
        ax.add_patch(arrow)
        if label:
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            ax.text(mx, my + 0.12, label, ha="center", va="bottom", fontsize=6, color="#666666")
