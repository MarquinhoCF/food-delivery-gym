import numpy as np

from food_delivery_gym.main.statistic.metrics.metric import Metric


class PoissonOrderGenerationMetric(Metric):

    def __init__(
        self,
        episode_events: list[dict] | None = None,
        rate_function=None,
        bin_size: float = 10,
    ):
        self.episode_events       = episode_events
        self.rate_function        = rate_function
        self.bin_size             = bin_size

    def view(self, ax) -> None:
        if self.episode_events is not None:
            self._view_single(ax)
        else:
            ax.text(0.5, 0.5, "Sem dados de eventos",
                    ha="center", va="center", transform=ax.transAxes)

    def _view_single(self, ax) -> None:
        arrival_times = [
            ev["time"] for ev in self.episode_events
            if ev["type"] == "CUSTOMER_PLACED_ORDER"
        ]
        if not arrival_times:
            ax.text(0.5, 0.5, "Nenhum pedido gerado",
                    ha="center", va="center", transform=ax.transAxes)
            return

        max_time = max(arrival_times)
        num_bins = int(np.ceil(max_time / self.bin_size))
        bins     = np.linspace(0, max_time, num_bins + 1)
        counts, bin_edges = np.histogram(arrival_times, bins=bins)
        bin_centers       = (bin_edges[:-1] + bin_edges[1:]) / 2

        ax.bar(
            bin_centers, counts,
            width=self.bin_size * 0.9,
            alpha=0.6, color="#3498db", edgecolor="#2980b9",
            linewidth=1, label="Observado",
        )

        title = "Geração de Pedidos"

        if self.rate_function is not None:
            expected = [self.rate_function(c) * self.bin_size for c in bin_centers]
            ax.plot(
                bin_centers, expected,
                color="#e74c3c", linewidth=3, linestyle="-",
                marker="o", markersize=6,
                label="Esperado (função de taxa)", alpha=0.8,
            )
            title += ": Observado vs Esperado"

        avg = np.mean(counts)
        ax.axhline(y=avg, color="#27ae60", linestyle="--", linewidth=2,
                   label=f"Média Obs. ({avg:.1f})", alpha=0.7)

        self._style(ax, title, "Tempo (min)", "Pedidos por Bin")

    @staticmethod
    def _style(ax, title, xlabel, ylabel):
        ax.set_title(title, fontsize=12, fontweight="bold", pad=15)
        ax.set_xlabel(xlabel, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
        ax.legend(loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        ax.set_ylim(bottom=0)
        ax.set_axisbelow(True)