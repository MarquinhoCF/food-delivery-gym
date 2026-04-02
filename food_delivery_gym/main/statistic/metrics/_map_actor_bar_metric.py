"""
Helper interno para métricas de barra por MapActor

Não importar diretamente — use as subclasses específicas.
"""

from matplotlib.ticker import FuncFormatter

from food_delivery_gym.main.statistic.metrics.metric import Metric


class _MapActorBarMetric(Metric):
    
    _metric_key        = ""             # chave para extrair a métrica do dicionário de dados do agente
    _bar_color         = "steelblue"    # cor para gráfico de barras no modo episódio único
    _xlabel            = ""             # label do eixo X (geralmente "ID do Motorista" ou "ID do Estabelecimento")
    _ylabel_single     = ""             # label do eixo Y para modo episódio único (ex: "Tempo Médio de Entrega (min)")
    _ylabel_agg        = ""             # label do eixo Y para modo agregado (ex: "Tempo Médio de Entrega (min)")
    _title_single      = ""             # título para modo episódio único
    _title_agg         = ""             # título para modo agregado
    _map_actor_label   = "Ator do Mapa" # label genérico para os agentes (substituído por "Motoristas" ou "Estabelecimentos" nas subclasses)

    def __init__(
        self,
        episode_data: dict | None = None,   # {agent_id: scalar_value}
        aggregate_data: dict | None = None, # {agent_id: {"avg": float, "std_dev": float, "median": float, "mode": float|str}}
    ):
        self.episode_data   = episode_data
        self.aggregate_data = aggregate_data

    def view(self, ax) -> None:
        if self.aggregate_data is not None:
            self._view_aggregate(ax)
        elif self.episode_data is not None:
            self._view_single(ax)
        else:
            ax.text(0.5, 0.5, "Sem dados",
                    ha="center", va="center", transform=ax.transAxes)

    def _view_single(self, ax) -> None:
        ids    = list(self.episode_data.keys())
        values = [self.episode_data[i] or 0 for i in ids]

        ax.barh(ids, values, color=self._bar_color)
        ax.set_xlabel(self._xlabel,        fontsize=11, fontweight="bold")
        ax.set_ylabel(self._map_actor_label,   fontsize=11, fontweight="bold")
        ax.set_title(self._title_single,   fontsize=12, fontweight="bold", pad=15)
        ax.set_yticks(range(len(ids)))
        ax.set_yticklabels([str(i) for i in ids])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))

    def _view_aggregate(self, ax) -> None:
        ids     = list(self.aggregate_data.keys())
        stats   = [self.aggregate_data[i] for i in ids]

        means   = [s["avg"]     if s else 0.0 for s in stats]
        medians = [s["median"]  if s else 0.0 for s in stats]
        std_dev = [s["std_dev"] if s else 0.0 for s in stats]
        modes   = [
            s["mode"] if s and isinstance(s["mode"], (int, float)) else 0.0
            for s in stats
        ]

        ax.errorbar(ids, means, yerr=std_dev, fmt="o", label="Média", capsize=5)
        ax.plot(ids, medians, marker="s", linestyle="", label="Mediana")
        ax.plot(ids, modes,   marker="^", linestyle="", label="Moda")

        ax.set_xlabel(self._map_actor_label,  fontsize=11, fontweight="bold")
        ax.set_ylabel(self._ylabel_agg,   fontsize=11, fontweight="bold")
        ax.set_title(self._title_agg,     fontsize=12, fontweight="bold", pad=15)
        ax.legend()
        ax.grid(True)