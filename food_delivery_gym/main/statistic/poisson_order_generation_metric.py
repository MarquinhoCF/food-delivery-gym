import numpy as np

from food_delivery_gym.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv
from food_delivery_gym.main.events.event_type import EventType
from food_delivery_gym.main.generator.non_homogeneous_poisson_order_generator import NonHomogeneousPoissonOrderGenerator
from food_delivery_gym.main.statistic.metric import Metric

class PoissonOrderGenerationMetric(Metric):
    """
    Métrica para comparar a distribuição observada com a esperada.
    
    Útil para validar se o gerador está funcionando corretamente,
    mostrando a distribuição teórica vs observada.
    
    Parameters:
    -----------
    environment : FoodDeliverySimpyEnv
        Ambiente de simulação
    rate_function : callable, optional
        Função de taxa esperada para comparação (apenas não-homogêneo)
    bin_size : float, optional
        Tamanho do bin temporal em minutos (default: 10)
    """
    
    def __init__(self, environment: FoodDeliverySimpyEnv, bin_size: float = 10):
        super().__init__(environment)
        self.bin_size = bin_size
    
    def view(self, ax) -> None:
        # Filtrar eventos de pedidos
        order_events = [
            event for event in self.environment.events
            if event.event_type == EventType.CUSTOMER_PLACED_ORDER
        ]
        
        if not order_events:
            ax.text(0.5, 0.5, 'No orders generated', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        arrival_times = [event.time for event in order_events]
        max_time = max(arrival_times)
        
        # Criar bins
        num_bins = int(np.ceil(max_time / self.bin_size))
        bins = np.linspace(0, max_time, num_bins + 1)
        
        # Distribuição observada
        counts, bin_edges = np.histogram(arrival_times, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Plotar distribuição observada
        ax.bar(
            bin_centers,
            counts,
            width=self.bin_size * 0.9,
            alpha=0.6,
            color='#3498db',
            label='Observed',
            edgecolor='#2980b9',
            linewidth=1
        )

        graph_title = 'Order Generation'
        
        # Plotar distribuição esperada se o gerador de Poisson for não homogêneo e a rate_function for fornecida
        non_homogeneous_poisson_generator = next(
            (generator for generator in self.environment.generators if isinstance(generator, NonHomogeneousPoissonOrderGenerator)),
            None
        )
        
        if non_homogeneous_poisson_generator is not None:
            rate_function = non_homogeneous_poisson_generator.get_rate_function()

            expected_counts = []
            for center in bin_centers:
                # Taxa média no bin
                rate = rate_function(center)
                expected = rate * self.bin_size
                expected_counts.append(expected)
            
            ax.plot(
                bin_centers,
                expected_counts,
                color='#e74c3c',
                linewidth=3,
                linestyle='-',
                marker='o',
                markersize=6,
                label='Expected (rate function)',
                alpha=0.8
            )

            graph_title += ': Observed vs Expected'
        
        # Linha da média observada
        avg_count = np.mean(counts)
        ax.axhline(
            y=avg_count,
            color='#27ae60',
            linestyle='--',
            linewidth=2,
            label=f'Observed Avg ({avg_count:.1f})',
            alpha=0.7
        )
        
        # Configurações
        ax.set_xlabel('Time (minutes)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Orders per Bin', fontsize=11, fontweight='bold')
        ax.set_title(
            graph_title,
            fontsize=12,
            fontweight='bold',
            pad=15
        )
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_ylim(bottom=0)
        ax.set_axisbelow(True)
