from collections import defaultdict

from food_delivery_gym.main.statistic.metrics.metric import Metric


class OrderFlowMetric(Metric):

    _PIPELINE = {
        "CUSTOMER_PLACED_ORDER":        ("Criado",       "#3498db", "o"),
        "ESTABLISHMENT_FINISHED_ORDER": ("Preparado",    "#f39c12", "s"),
        "DRIVER_PICKED_UP_ORDER":       ("Coletado",     "#9b59b6", "^"),
        "DRIVER_DELIVERED_ORDER":       ("Entregue",     "#27ae60", "D"),
    }

    def __init__(
        self,
        episode_events: list[dict] | None = None,
    ):
        self.episode_events      = episode_events

    def view(self, ax) -> None:
        if self.episode_events is not None:
            self._view_single(ax)
        else:
            ax.text(0.5, 0.5, "Sem dados de eventos",
                    ha="center", va="center", transform=ax.transAxes)

    def _view_single(self, ax) -> None:
        event_times: dict[str, list] = defaultdict(list)
        for ev in self.episode_events:
            if ev["type"] in self._PIPELINE:
                event_times[ev["type"]].append(ev["time"])

        if not any(event_times.values()):
            ax.text(0.5, 0.5, "Sem eventos de pedido",
                    ha="center", va="center", transform=ax.transAxes)
            return

        for etype, (label, color, marker) in self._PIPELINE.items():
            times = sorted(event_times[etype])
            if not times:
                continue
            cumulative = list(range(1, len(times) + 1))
            ax.plot(
                times, cumulative,
                label=label, color=color, linewidth=2.5,
                marker=marker, markevery=max(1, len(times) // 15),
                markersize=6, alpha=0.85,
            )

        placed    = len(event_times["CUSTOMER_PLACED_ORDER"])
        delivered = len(event_times["DRIVER_DELIVERED_ORDER"])
        if placed > 0:
            pct = delivered / placed * 100
            txt = f"Total: {placed}\nEntregues: {delivered} ({pct:.1f}%)"
            ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5), fontsize=9)

        self._style(ax, "Fluxo de Pedidos – Visão Acumulada")

    @staticmethod
    def _style(ax, title):
        ax.set_title(title, fontsize=12, fontweight="bold", pad=15)
        ax.set_xlabel("Tempo (min)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Contagem Acumulada", fontsize=11, fontweight="bold")
        ax.legend(loc="upper left", framealpha=0.9, fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        ax.set_ylim(bottom=0)
        ax.set_axisbelow(True)