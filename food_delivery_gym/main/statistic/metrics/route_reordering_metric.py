import numpy as np

from food_delivery_gym.main.statistic.metrics.metric import Metric


class RouteReorderingMetric(Metric):
    """
    Estatísticas de reordenação de rotas.

    Modos de uso:
    ─────────────
    - Episódio único  : passe ``episode_drivers``
      episode_drivers = {"0": {"reordering_total_reorderings": 3,
                               "reordering_net_time_impact": 10.5, ...}, ...}

    - Agregado        : passe ``aggregate_drivers``
      aggregate_drivers = {"0": {"reordering_total_reorderings": {"avg": 3, "std_dev": 0.5, ...},
                                 "reordering_net_time_impact":   {"avg": 9, ...}, ...}, ...}

    As chaves de reordenação esperadas são as produzidas por SimulationStats
    ao achatar dicts aninhados: ``reordering_<sub_key>``.
    """

    _REORDER_KEYS = [
        "reordering_total_reorderings",
        "reordering_successful_reorderings",
        "reordering_failed_reorderings",
        "reordering_net_time_impact",
        "reordering_net_distance_impact",
        "reordering_success_rate",
    ]

    def __init__(
        self,
        episode_drivers: dict | None = None,
        aggregate_drivers: dict | None = None,
    ):
        self.episode_drivers   = episode_drivers
        self.aggregate_drivers = aggregate_drivers

    # ─────────────────────────────────────────────────────────────────
    def view(self, ax) -> None:
        if self.aggregate_drivers is not None:
            self._view_aggregate(ax)
        elif self.episode_drivers is not None:
            self._view_single(ax)
        else:
            ax.text(0.5, 0.5, "Sem dados de reordenação",
                    ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")

    # ─────────────────────────────────────────────────────────────────
    def _view_single(self, ax) -> None:
        driver_ids = list(self.episode_drivers.keys())

        def _get(did, key, default=0):
            return self.episode_drivers[did].get(key, default) or default

        successful     = [_get(d, "reordering_successful_reorderings") for d in driver_ids]
        failed         = [_get(d, "reordering_failed_reorderings")     for d in driver_ids]
        net_time       = [_get(d, "reordering_net_time_impact")        for d in driver_ids]
        net_distance   = [_get(d, "reordering_net_distance_impact")    for d in driver_ids]
        success_rate   = [_get(d, "reordering_success_rate")           for d in driver_ids]

        self._render_4panel(ax, driver_ids, successful, failed,
                            net_time, net_distance, success_rate)

    # ─────────────────────────────────────────────────────────────────
    def _view_aggregate(self, ax) -> None:
        driver_ids = list(self.aggregate_drivers.keys())

        def _avg(did, key):
            s = self.aggregate_drivers[did].get(key)
            return s["avg"] if s else 0.0

        successful   = [_avg(d, "reordering_successful_reorderings") for d in driver_ids]
        failed       = [_avg(d, "reordering_failed_reorderings")     for d in driver_ids]
        net_time     = [_avg(d, "reordering_net_time_impact")        for d in driver_ids]
        net_distance = [_avg(d, "reordering_net_distance_impact")    for d in driver_ids]
        success_rate = [_avg(d, "reordering_success_rate")           for d in driver_ids]

        self._render_4panel(ax, driver_ids, successful, failed,
                            net_time, net_distance, success_rate,
                            title_suffix=" (Média)")

    # ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _render_4panel(ax, driver_ids, successful, failed,
                       net_time, net_distance, success_rate,
                       title_suffix=""):
        fig = ax.get_figure()
        ax.clear()
        ax.axis("off")
        gs = fig.add_gridspec(2, 2, hspace=1.2, wspace=1.2)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])

        x     = np.arange(len(driver_ids))
        width = 0.35

        # — Painel 1: Sucesso vs Falha ——————————————————————————————
        ax1.bar(x - width / 2, successful, width, label="Bem-sucedidas",
                color="green", alpha=0.7, edgecolor="black")
        ax1.bar(x + width / 2, failed, width, label="Falhas",
                color="red", alpha=0.7, edgecolor="black")
        ax1.set_xticks(x)
        ax1.set_xticklabels([str(d) for d in driver_ids])
        ax1.set_xlabel("Motoristas", fontsize=10, fontweight="bold")
        ax1.set_ylabel("Reordenações", fontsize=10, fontweight="bold")
        ax1.set_title(f"Sucesso vs Falha{title_suffix}", fontsize=11, fontweight="bold")
        ax1.legend(); ax1.grid(True, alpha=0.3, axis="y")

        # — Painel 2: Impacto em Tempo ——————————————————————————————
        colors_t = ["green" if v > 0 else "red" for v in net_time]
        ax2.barh([str(d) for d in driver_ids], net_time,
                 color=colors_t, alpha=0.7, edgecolor="black")
        ax2.axvline(x=0, color="black", linestyle="--", linewidth=1.5)
        ax2.set_ylabel("Motoristas", fontsize=10, fontweight="bold")
        ax2.set_xlabel("Tempo (un.)", fontsize=9, fontweight="bold")
        ax2.set_title(f"Impacto em Tempo{title_suffix}\n(+) Economizado / (-) Perdido",
                      fontsize=11, fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="x")
        for i, v in enumerate(net_time):
            ax2.text(v, i, f" {v:.1f}", va="center",
                     ha="left" if v >= 0 else "right", fontsize=8)

        # — Painel 3: Impacto em Distância ——————————————————————————
        colors_d = ["blue" if v > 0 else "orange" for v in net_distance]
        ax3.barh([str(d) for d in driver_ids], net_distance,
                 color=colors_d, alpha=0.7, edgecolor="black")
        ax3.axvline(x=0, color="black", linestyle="--", linewidth=1.5)
        ax3.set_ylabel("Motoristas", fontsize=10, fontweight="bold")
        ax3.set_xlabel("Distância (un.)", fontsize=9, fontweight="bold")
        ax3.set_title(f"Impacto em Distância{title_suffix}\n(+) Economizada / (-) Aumentada",
                      fontsize=11, fontweight="bold")
        ax3.grid(True, alpha=0.3, axis="x")
        for i, v in enumerate(net_distance):
            ax3.text(v, i, f" {v:.1f}", va="center",
                     ha="left" if v >= 0 else "right", fontsize=8)

        # — Painel 4: Taxa de Sucesso ————————————————————————————————
        ax4.bar([str(d) for d in driver_ids], success_rate,
                color="skyblue", alpha=0.7, edgecolor="black")
        ax4.axhline(y=50, color="orange", linestyle="--", linewidth=1.5, label="50% ref.")
        ax4.set_ylim(0, 105)
        ax4.set_xlabel("Motoristas", fontsize=10, fontweight="bold")
        ax4.set_ylabel("Taxa de Sucesso (%)", fontsize=10, fontweight="bold")
        ax4.set_title(f"Taxa de Sucesso{title_suffix}", fontsize=11, fontweight="bold")
        ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3, axis="y")
        for i, rate in enumerate(success_rate):
            ax4.text(i, rate + 2, f"{rate:.1f}%", ha="center", fontsize=8)