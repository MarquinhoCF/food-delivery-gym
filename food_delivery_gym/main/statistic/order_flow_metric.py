from collections import defaultdict

from food_delivery_gym.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv
from food_delivery_gym.main.events.event_type import EventType
from food_delivery_gym.main.statistic.metric import Metric


class OrderFlowMetric(Metric):
    """
    Métrica para visualizar o fluxo completo de pedidos ao longo do tempo.
    
    Mostra curvas acumuladas dos principais eventos do pipeline:
    - Pedidos criados (CUSTOMER_PLACED_ORDER)
    - Pedidos finalizados pelo estabelecimento (ESTABLISHMENT_FINISHED_ORDER)
    - Pedidos coletados pelo entregador (DRIVER_PICKED_UP_ORDER)
    - Pedidos entregues (DRIVER_DELIVERED_ORDER)
    
    Parameters:
    -----------
    environment : FoodDeliverySimpyEnv
        Ambiente de simulação
    show_rates : bool, optional
        Se True, mostra também as taxas instantâneas (default: False)
    bin_size : float, optional
        Tamanho do bin para calcular taxas em minutos (default: 5)
    """
    
    def __init__(self, environment: FoodDeliverySimpyEnv, 
                 show_rates: bool = False, 
                 bin_size: float = 5):
        super().__init__(environment)
        self.show_rates = show_rates
        self.bin_size = bin_size
    
    def view(self, ax) -> None:
        # Definir os eventos relevantes no pipeline
        pipeline_events = {
            EventType.CUSTOMER_PLACED_ORDER: {
                'label': 'Placed',
                'color': '#3498db',
                'marker': 'o'
            },
            EventType.ESTABLISHMENT_FINISHED_ORDER: {
                'label': 'Prepared',
                'color': '#f39c12',
                'marker': 's'
            },
            EventType.DRIVER_PICKED_UP_ORDER: {
                'label': 'Picked Up',
                'color': '#9b59b6',
                'marker': '^'
            },
            EventType.DRIVER_DELIVERED_ORDER: {
                'label': 'Delivered',
                'color': '#27ae60',
                'marker': 'D'
            }
        }
        
        # Coletar timestamps por tipo de evento
        event_times = defaultdict(list)
        for event in self.environment.events:
            if event.event_type in pipeline_events:
                event_times[event.event_type].append(event.time)
        
        # Verificar se há dados
        if not any(event_times.values()):
            ax.text(0.5, 0.5, 'No order events found', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Plotar curvas acumuladas
        for event_type, config in pipeline_events.items():
            times = sorted(event_times[event_type])
            if not times:
                continue
                
            # Criar curva acumulada
            cumulative = list(range(1, len(times) + 1))
            
            ax.plot(
                times,
                cumulative,
                label=config['label'],
                color=config['color'],
                linewidth=2.5,
                marker=config['marker'],
                markevery=max(1, len(times) // 15),  # Mostrar ~15 marcadores
                markersize=6,
                alpha=0.85
            )
        
        # Adicionar informações estatísticas
        total_placed = len(event_times[EventType.CUSTOMER_PLACED_ORDER])
        total_delivered = len(event_times[EventType.DRIVER_DELIVERED_ORDER])
        
        if total_placed > 0:
            completion_rate = (total_delivered / total_placed) * 100
            
            # Adicionar texto com estatísticas
            stats_text = f'Total Orders: {total_placed}\n'
            stats_text += f'Delivered: {total_delivered} ({completion_rate:.1f}%)'
            
            ax.text(
                0.02, 0.98, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9
            )
        
        # Configurações do gráfico
        ax.set_xlabel('Time (minutes)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Cumulative Count', fontsize=11, fontweight='bold')
        ax.set_title(
            'Order Flow Pipeline - Cumulative View',
            fontsize=12,
            fontweight='bold',
            pad=15
        )
        ax.legend(loc='upper left', framealpha=0.9, fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_ylim(bottom=0)
        ax.set_axisbelow(True)