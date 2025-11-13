from collections import defaultdict
import numpy as np
from food_delivery_gym.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv
from food_delivery_gym.main.events.event_type import EventType
from food_delivery_gym.main.statistic.metric import Metric


class OrderPipelineStatusMetric(Metric):
    """
    Métrica para visualizar o status dos pedidos em cada estágio do pipeline.
    
    Mostra quantos pedidos estão em cada estado ao longo do tempo:
    - Aguardando preparo
    - Em preparo
    - Aguardando coleta
    - Em trânsito
    - Entregues
    
    A legenda é interativa: clique nos itens para mostrar/ocultar estados.

    Essa métrica é muito pesada!!
    
    Parameters:
    -----------
    environment : FoodDeliverySimpyEnv
        Ambiente de simulação
    time_step : float, optional
        Intervalo de tempo para amostrar o estado (default: 1 minuto)
    """
    
    def __init__(self, environment: FoodDeliverySimpyEnv, time_step: float = 1.0):
        super().__init__(environment)
        self.time_step = time_step
        self.visible_states = {}  # Armazena estado de visibilidade
    
    def view(self, ax) -> None:
        # Coletar todos os eventos relevantes
        order_events = defaultdict(list)
        
        for event in self.environment.events:
            if hasattr(event, 'order') and hasattr(event.order, 'order_id'):
                order_events[event.order.order_id].append(event)
        
        if not order_events:
            ax.text(0.5, 0.5, 'No order data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Determinar tempo máximo
        max_time = max(event.time for event in self.environment.events)
        time_points = np.arange(0, max_time, self.time_step)
        
        # Estados possíveis
        states = {
            'waiting_prep': [],
            'preparing': [],
            'waiting_pickup': [],
            'in_transit': [],
            'delivered': []
        }
        
        # Para cada ponto no tempo, contar pedidos em cada estado
        for t in time_points:
            counts = {state: 0 for state in states}
            
            for order_id, events in order_events.items():
                state = self._get_order_state_at_time(events, t)
                if state:
                    counts[state] += 1
            
            for state in states:
                states[state].append(counts[state])
        
        # Plotar área empilhada
        colors = {
            'waiting_prep': '#e74c3c',
            'preparing': '#f39c12',
            'waiting_pickup': '#3498db',
            'in_transit': '#9b59b6',
            'delivered': '#27ae60'
        }
        
        labels = {
            'waiting_prep': 'Waiting Prep',
            'preparing': 'Preparing',
            'waiting_pickup': 'Waiting Pickup',
            'in_transit': 'In Transit',
            'delivered': 'Delivered'
        }
        
        # Armazenar dados para redesenho
        self.time_points = time_points
        self.states_data = states
        self.colors = colors
        self.labels = labels
        self.ax = ax
        
        # Inicializar todos os estados como visíveis
        for state in states.keys():
            self.visible_states[state] = True
        
        # Desenhar gráfico inicial
        self._draw_chart()
        
        # Configurar interatividade da legenda
        legend = ax.get_legend()
        if legend:
            for legend_line in legend.get_patches():
                legend_line.set_picker(True)
            
            # Conectar evento de clique
            ax.figure.canvas.mpl_connect('pick_event', self._on_legend_click)
    
    def _draw_chart(self):
        """Desenha ou redesenha o gráfico com base nos estados visíveis."""
        self.ax.clear()
        
        # Criar stacked area chart apenas com estados visíveis
        y_stack = np.zeros(len(self.time_points))
        
        for state in ['waiting_prep', 'preparing', 'waiting_pickup', 'in_transit', 'delivered']:
            if self.visible_states[state]:
                self.ax.fill_between(
                    self.time_points,
                    y_stack,
                    y_stack + self.states_data[state],
                    label=self.labels[state],
                    color=self.colors[state],
                    alpha=0.7
                )
                y_stack += self.states_data[state]
        
        # Configurações
        self.ax.set_xlabel('Time (minutes)', fontsize=11, fontweight='bold')
        self.ax.set_ylabel('Number of Orders', fontsize=11, fontweight='bold')
        self.ax.set_title(
            'Order Status Distribution Over Time',
            fontsize=12,
            fontweight='bold',
            pad=15
        )
        self.ax.legend(loc='upper left', framealpha=0.9, fontsize=9)
        self.ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        self.ax.set_ylim(bottom=0)
        self.ax.set_axisbelow(True)
        
        # Configurar interatividade novamente após redesenhar
        legend = self.ax.get_legend()
        if legend:
            for legend_line in legend.get_patches():
                legend_line.set_picker(True)
    
    def _on_legend_click(self, event):
        """Callback para clique na legenda."""
        legend = self.ax.get_legend()
        if not legend:
            return
        
        # Encontrar qual item da legenda foi clicado
        patches = legend.get_patches()
        state_order = ['waiting_prep', 'preparing', 'waiting_pickup', 'in_transit', 'delivered']
        
        for idx, patch in enumerate(patches):
            if event.artist == patch:
                # Alternar visibilidade do estado correspondente
                state = state_order[idx]
                self.visible_states[state] = not self.visible_states[state]
                
                # Atualizar aparência da legenda (alpha reduzido quando invisível)
                if self.visible_states[state]:
                    patch.set_alpha(0.7)
                else:
                    patch.set_alpha(0.2)
                
                # Redesenhar gráfico
                self._draw_chart()
                self.ax.figure.canvas.draw()
                break
    
    def _get_order_state_at_time(self, events, time):
        """Determina o estado de um pedido em um dado momento."""
        # Ordenar eventos por tempo
        sorted_events = sorted(events, key=lambda e: e.time)
        
        # Encontrar o último evento antes do tempo especificado
        relevant_events = [e for e in sorted_events if e.time <= time]
        
        if not relevant_events:
            return None
        
        last_event = relevant_events[-1]
        
        # Mapear tipo de evento para estado
        event_to_state = {
            EventType.CUSTOMER_PLACED_ORDER: 'waiting_prep',
            EventType.ESTABLISHMENT_ACCEPTED_ORDER: 'preparing',
            EventType.ESTABLISHMENT_FINISHED_ORDER: 'waiting_pickup',
            EventType.DRIVER_PICKED_UP_ORDER: 'in_transit',
            EventType.DRIVER_DELIVERED_ORDER: 'delivered'
        }
        
        return event_to_state.get(last_event.event_type)