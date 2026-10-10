from __future__ import annotations

import json
import os
from typing import Any

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from food_delivery_gym.main.statistics.boards.board import Board


class MCTSDecisionBoard(Board):
    """
    Visualização pós-episódio das decisões do MonteCarloTreeSearchOptimizerGym.

    Figura 1x2:
      - esquerda: árvore explorada (raiz → ação → outcome → filho)
      - direita: barras immediate_reward + α·continuation (= action_value)
    """

    _NODE_W = 1.45
    _NODE_H = 0.78
    _X_GAP = 2.0
    _Y_GAP = 1.05

    def __init__(self, decision_log: list[dict]) -> None:
        super().__init__(metrics=[])
        self.decision_log = decision_log

    def view(self) -> None:
        for decision in self.decision_log:
            fig = self._build_figure(decision)
            plt.show()
            plt.close(fig)

    def save(self, dir_path: str, decision_idx: int) -> None:
        matplotlib.use("Agg")
        decision = self._require_decision(decision_idx)
        out_dir = os.path.join(dir_path, "mcts_decisions")
        os.makedirs(out_dir, exist_ok=True)
        fig = self._build_figure(decision)
        name = f"decision_{decision_idx + 1:03d}.png"
        fig.savefig(os.path.join(out_dir, name), dpi=150, bbox_inches="tight")
        plt.close(fig)

    def dump_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.decision_log, f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: str) -> "MCTSDecisionBoard":
        with open(path, "r", encoding="utf-8") as f:
            decision_log = json.load(f)
        return cls(decision_log)

    def _require_decision(self, decision_idx: int) -> dict[str, Any]:
        n = len(self.decision_log)
        if n == 0:
            raise SystemExit("decision_log vazio: nenhuma decisão gravada")
        if decision_idx < 0 or decision_idx >= n:
            raise SystemExit(
                f"--decision={decision_idx} fora do range; disponíveis: 0..{n - 1}"
            )
        for decision in self.decision_log:
            if int(decision.get("decision_idx", -1)) == decision_idx:
                return decision
        return self.decision_log[decision_idx]

    def _build_figure(self, decision: dict[str, Any]) -> Figure:
        candidates = decision.get("candidates") or []
        tree = decision.get("tree") or {"actions": [], "visit_count": 0, "depth": 0}
        n_cand = max(len(candidates), 1)
        leaf_slots = max(self._subtree_slots(tree), 1)
        tree_depth = max(self._tree_depth(tree), 1)
        fig_w = max(14.0, 4.0 + tree_depth * 2.2)
        fig_h = max(6.0, 2.5 + max(n_cand, leaf_slots) * 0.95)

        fig, (ax_tree, ax_bars) = plt.subplots(
            1, 2, figsize=(fig_w, fig_h), gridspec_kw={"width_ratios": [2.4, 1.0]}
        )
        fig.suptitle(
            f"Decision {int(decision.get('decision_idx', 0)) + 1}  |  "
            f"order_id={decision.get('order_id')}  |  "
            f"sim_time={decision.get('sim_time')}  |  "
            f"i={decision.get('iterations')}  d={decision.get('depth')}  "
            f"ew={decision.get('exploration_weight')}  "
            f"maxexp={'all' if decision.get('max_expanded_actions') is None else decision.get('max_expanded_actions')}",
            fontsize=12,
            fontweight="bold",
        )

        self._draw_tree(ax_tree, decision, tree)
        self._draw_action_bars(
            ax_bars,
            candidates,
            decision.get("chosen_action"),
            float(decision.get("alpha", 1.0)),
        )
        fig.tight_layout()
        return fig

    def _subtree_slots(self, node: dict | None) -> int:
        if not node:
            return 1
        actions = node.get("actions") or []
        if not actions:
            return 1
        total = 0
        for action in actions:
            outcomes = action.get("outcomes") or []
            if not outcomes:
                total += 1
            else:
                for outcome in outcomes:
                    total += self._subtree_slots(outcome.get("node"))
        return max(total, 1)

    def _tree_depth(self, node: dict | None) -> int:
        if not node:
            return 0
        actions = node.get("actions") or []
        if not actions:
            return 1
        best = 1
        for action in actions:
            for outcome in action.get("outcomes") or []:
                # action + outcome + child
                best = max(best, 2 + self._tree_depth(outcome.get("node")))
        return best

    def _draw_action_bars(
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

        immediate = [c["immediate_reward"] for c in candidates]
        continuation = [
            alpha * float(c.get("continuation_value", 0.0)) for c in candidates
        ]
        action_values = [c["action_value"] for c in candidates]
        labels = [
            f"a={c['action']}\nid={c['driver_id']}\nn={c.get('action_visits', 0)}"
            for c in candidates
        ]
        x = range(len(candidates))
        width = 0.55

        bars_imm = ax.bar(x, immediate, width, label="immediate_reward", color="#4C78A8")
        bars_cont = ax.bar(
            x,
            continuation,
            width,
            bottom=immediate,
            label=f"α·continuation (α={alpha:g})",
            color="#F58518",
        )

        for i, (imm_bar, cont_bar, q) in enumerate(
            zip(bars_imm, bars_cont, action_values)
        ):
            if candidates[i]["action"] == chosen_action:
                for bar in (imm_bar, cont_bar):
                    bar.set_edgecolor("black")
                    bar.set_linewidth(2.5)
                    bar.set_hatch("//")
            top = imm_bar.get_height() + cont_bar.get_height()
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
        ax.set_ylabel("action_value components")
        ax.set_title("Root action values (stacked)")
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.legend(loc="best", fontsize=8)
        ax.grid(axis="y", linestyle=":", alpha=0.5)

    def _draw_tree(self, ax, decision: dict, tree: dict) -> None:
        ax.set_title("MCTS explored tree (root → action → outcome → child)")
        ax.set_aspect("equal")
        ax.axis("off")

        if not (tree.get("actions") or []):
            ax.text(
                0.5,
                0.5,
                "Empty tree",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return

        chosen_action = decision.get("chosen_action")
        positions: list[tuple[float, float]] = []
        slots = self._subtree_slots(tree)
        y_top = (slots - 1) * self._Y_GAP
        root_xy = (0.0, y_top / 2.0)
        root_label = (
            f"order {decision.get('order_id')}\n"
            f"t={decision.get('sim_time')}\n"
            f"visits={tree.get('visit_count', 0)}"
        )
        self._draw_node(ax, root_xy, root_label, face="#E8E8E8", edge="#333333", lw=1.5)
        positions.append(root_xy)

        self._layout_children(
            ax,
            tree,
            root_xy,
            x_level=1,
            y_start=0.0,
            chosen_action=chosen_action,
            is_root=True,
            positions=positions,
        )

        all_x = [p[0] for p in positions]
        all_y = [p[1] for p in positions]
        pad_x = self._NODE_W
        pad_y = self._NODE_H
        ax.set_xlim(min(all_x) - pad_x, max(all_x) + pad_x)
        ax.set_ylim(min(all_y) - pad_y, max(all_y) + pad_y)
        ax.text(
            0.01,
            0.01,
            "Green = chosen action  |  Blue = action  |  Orange = outcome  |  Gray = node",
            transform=ax.transAxes,
            fontsize=8,
            color="#555555",
            va="bottom",
        )

    def _layout_children(
        self,
        ax,
        node: dict,
        parent_xy: tuple[float, float],
        x_level: int,
        y_start: float,
        chosen_action: int | None,
        is_root: bool,
        positions: list[tuple[float, float]],
    ) -> float:
        """Desenha ações/outcomes/filhos; retorna y final usado."""
        y_cursor = y_start
        for action in sorted(node.get("actions") or [], key=lambda a: a["action"]):
            outcomes = action.get("outcomes") or []
            action_slots = (
                sum(self._subtree_slots(o.get("node")) for o in outcomes)
                if outcomes
                else 1
            )
            action_h = max(action_slots, 1) * self._Y_GAP
            action_xy = (
                x_level * self._X_GAP,
                y_cursor + (action_h - self._Y_GAP) / 2.0,
            )
            is_chosen = is_root and action["action"] == chosen_action
            face = "#C7E9C0" if is_chosen else "#DEEBF7"
            edge = "#006D2C" if is_chosen else "#3182BD"
            lw = 2.4 if is_chosen else 1.2
            label = (
                f"a={action['action']} drv={action['driver_id']}\n"
                f"r₀={action['immediate_reward']:.2f}\n"
                f"Q={action['action_value']:.2f} n={action['action_visits']}"
            )
            self._draw_node(ax, action_xy, label, face=face, edge=edge, lw=lw)
            self._draw_edge(ax, parent_xy, action_xy, label=f"a={action['action']}")
            positions.append(action_xy)

            if not outcomes:
                y_cursor += self._Y_GAP
                continue

            outcome_y = y_cursor
            for outcome in outcomes:
                child = outcome.get("node")
                child_slots = self._subtree_slots(child)
                child_h = max(child_slots, 1) * self._Y_GAP
                outcome_xy = (
                    (x_level + 1) * self._X_GAP,
                    outcome_y + (child_h - self._Y_GAP) / 2.0,
                )
                out_label = f"seed={outcome.get('scenario_seed')}"
                self._draw_node(
                    ax,
                    outcome_xy,
                    out_label,
                    face="#FFF5EB",
                    edge="#D94801",
                    lw=1.1,
                )
                self._draw_edge(ax, action_xy, outcome_xy, label="W")
                positions.append(outcome_xy)

                if child is not None:
                    child_xy = (
                        (x_level + 2) * self._X_GAP,
                        outcome_y + (child_h - self._Y_GAP) / 2.0,
                    )
                    done = child.get("done") or child.get("truncated")
                    child_label = (
                        f"d={child.get('depth')} "
                        f"{'end' if done else 'node'}\n"
                        f"visits={child.get('visit_count', 0)}"
                    )
                    self._draw_node(
                        ax,
                        child_xy,
                        child_label,
                        face="#FEE0D2" if done else "#F0F0F0",
                        edge="#A63603" if done else "#666666",
                        lw=1.2,
                    )
                    self._draw_edge(ax, outcome_xy, child_xy, label="")
                    positions.append(child_xy)
                    if child.get("actions"):
                        self._layout_children(
                            ax,
                            child,
                            child_xy,
                            x_level=x_level + 3,
                            y_start=outcome_y,
                            chosen_action=None,
                            is_root=False,
                            positions=positions,
                        )
                outcome_y += child_h

            y_cursor += action_h
        return y_cursor

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
            ax.text(
                mx,
                my + 0.12,
                label,
                ha="center",
                va="bottom",
                fontsize=6,
                color="#666666",
            )
