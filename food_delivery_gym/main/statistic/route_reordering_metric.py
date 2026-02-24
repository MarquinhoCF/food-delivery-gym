import numpy as np
from food_delivery_gym.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv
from food_delivery_gym.main.statistic.metric import Metric


class RouteReorderingMetric(Metric):
    
    def __init__(self, environment: FoodDeliverySimpyEnv, drivers_statistics=None):
        super().__init__(environment)
        self.drivers_statistics = drivers_statistics
    
    def view(self, ax) -> None:
        if self.drivers_statistics is not None:
            self._view_with_statistics(ax)
        else:
            self._view_current_simulation(ax)
    
    def _view_with_statistics(self, ax):

        driver_ids = list(self.drivers_statistics.keys())
        
        # Extrai dados estatísticos
        total_reorderings_mean = [
            self.drivers_statistics[d]['reordering']['total_reorderings']['mean'] 
            for d in driver_ids
        ]
        net_impact_mean = [
            self.drivers_statistics[d]['reordering']['net_time_impact']['mean'] 
            for d in driver_ids
        ]
        net_distance_mean = [
            self.drivers_statistics[d]['reordering']['net_distance_impact']['mean'] 
            for d in driver_ids
        ]
        success_rate_mean = [
            self.drivers_statistics[d]['reordering']['success_rate']['mean'] 
            for d in driver_ids
        ]
        
        # Cria subplots em grid 2x2
        fig = ax.get_figure()
        ax.clear()
        gs = fig.add_gridspec(2, 2, hspace=1.2, wspace=1.2)
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Gráfico 1: Total de reordenações
        colors = ['green' if impact > 0 else 'red' for impact in net_impact_mean]
        ax1.bar(driver_ids, total_reorderings_mean, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Motoristas', fontsize=10, fontweight='bold')
        ax1.set_ylabel('Número de Reordenações', fontsize=10, fontweight='bold')
        ax1.set_title('Total de Reordenações por Motorista', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: Impacto líquido em tempo
        colors_impact = ['green' if x > 0 else 'red' for x in net_impact_mean]
        ax2.barh(driver_ids, net_impact_mean, color=colors_impact, alpha=0.7, edgecolor='black')
        ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax2.set_ylabel('Motoristas', fontsize=10, fontweight='bold')
        ax2.set_xlabel('Tempo Economizado (+) / Perdido (-)', fontsize=9, fontweight='bold')
        ax2.set_title('Impacto em Tempo', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Gráfico 3: Impacto líquido em distância
        colors_distance = ['blue' if x > 0 else 'orange' for x in net_distance_mean]
        ax3.barh(driver_ids, net_distance_mean, color=colors_distance, alpha=0.7, edgecolor='black')
        ax3.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax3.set_ylabel('Motoristas', fontsize=10, fontweight='bold')
        ax3.set_xlabel('Distância Economizada (+) / Aumentada (-)', fontsize=9, fontweight='bold')
        ax3.set_title('Impacto em Distância', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Gráfico 4: Taxa de sucesso
        ax4.bar(driver_ids, success_rate_mean, color='skyblue', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Motoristas', fontsize=10, fontweight='bold')
        ax4.set_ylabel('Taxa de Sucesso (%)', fontsize=10, fontweight='bold')
        ax4.set_title('Taxa de Sucesso das Reordenações', fontsize=11, fontweight='bold')
        ax4.set_ylim(0, 100)
        ax4.grid(True, alpha=0.3)
        
        # Remove o eixo original
        ax.axis('off')
    
    def _view_current_simulation(self, ax):
        """Visualiza dados da simulação atual"""
        drivers = self.environment.state.drivers
        
        # Filtra apenas DynamicRouteDrivers
        dynamic_drivers = [
            d for d in drivers 
            if hasattr(d, 'get_reordering_statistics')
        ]
        
        if not dynamic_drivers:
            ax.text(0.5, 0.5, 'Nenhum DynamicRouteDriver encontrado', 
                   ha='center', va='center', fontsize=12)
            ax.axis('off')
            return
        
        # Coleta estatísticas
        driver_ids = [d.driver_id for d in dynamic_drivers]
        stats = [d.get_reordering_statistics() for d in dynamic_drivers]
        
        total_reorderings = [s['total_reorderings'] for s in stats]
        successful = [s['successful_reorderings'] for s in stats]
        failed = [s['failed_reorderings'] for s in stats]
        net_time_impact = [s['net_time_impact'] for s in stats]
        net_distance_impact = [s['net_distance_impact'] for s in stats]
        success_rate = [s['success_rate'] for s in stats]
        
        # Cria visualização com múltiplos gráficos (2x2)
        fig = ax.get_figure()
        ax.clear()
        gs = fig.add_gridspec(2, 2, hspace=1.2, wspace=1.2)
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Gráfico 1: Reordenações bem-sucedidas vs falhas
        x = np.arange(len(driver_ids))
        width = 0.35
        
        ax1.bar(x - width/2, successful, width, label='Bem-sucedidas', 
               color='green', alpha=0.7, edgecolor='black')
        ax1.bar(x + width/2, failed, width, label='Falhas', 
               color='red', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Motoristas', fontsize=10, fontweight='bold')
        ax1.set_ylabel('Número de Reordenações', fontsize=10, fontweight='bold')
        ax1.set_title('Reordenações: Sucesso vs Falha', fontsize=11, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([str(int(d_id)) for d_id in driver_ids])
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Gráfico 2: Impacto líquido em tempo
        colors_time = ['green' if x > 0 else 'red' for x in net_time_impact]
        ax2.barh([str(int(d_id)) for d_id in driver_ids], net_time_impact, 
                color=colors_time, alpha=0.7, edgecolor='black')
        ax2.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
        ax2.set_ylabel('Motoristas', fontsize=10, fontweight='bold')
        ax2.set_xlabel('Tempo (unidades)', fontsize=9, fontweight='bold')
        ax2.set_title('Impacto em Tempo\nEconomizado (+) / Perdido (-)', 
                     fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Adiciona valores nas barras
        for i, (d_id, value) in enumerate(zip(driver_ids, net_time_impact)):
            ax2.text(value, i, f' {value:.1f}', 
                    va='center', ha='left' if value > 0 else 'right',
                    fontsize=8, fontweight='bold')
        
        # Gráfico 3: Impacto líquido em distância
        colors_distance = ['blue' if x > 0 else 'orange' for x in net_distance_impact]
        ax3.barh([str(int(d_id)) for d_id in driver_ids], net_distance_impact, 
                color=colors_distance, alpha=0.7, edgecolor='black')
        ax3.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
        ax3.set_ylabel('Motoristas', fontsize=10, fontweight='bold')
        ax3.set_xlabel('Distância (unidades)', fontsize=9, fontweight='bold')
        ax3.set_title('Impacto em Distância\nEconomizada (+) / Aumentada (-)', 
                     fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')
        
        # Adiciona valores nas barras
        for i, (d_id, value) in enumerate(zip(driver_ids, net_distance_impact)):
            ax3.text(value, i, f' {value:.1f}', 
                    va='center', ha='left' if value > 0 else 'right',
                    fontsize=8, fontweight='bold')
        
        # Gráfico 4: Taxa de sucesso com linha de referência
        ax4.bar([str(int(d_id)) for d_id in driver_ids], success_rate, 
               color='skyblue', alpha=0.7, edgecolor='black')
        ax4.axhline(y=50, color='orange', linestyle='--', linewidth=1.5, 
                   label='50% de referência')
        ax4.set_xlabel('Motoristas', fontsize=10, fontweight='bold')
        ax4.set_ylabel('Taxa de Sucesso (%)', fontsize=10, fontweight='bold')
        ax4.set_title('Taxa de Sucesso das Reordenações', fontsize=11, fontweight='bold')
        ax4.set_ylim(0, 105)
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Adiciona valores nas barras
        for i, (d_id, rate) in enumerate(zip(driver_ids, success_rate)):
            ax4.text(i, rate + 2, f'{rate:.1f}%', 
                    ha='center', fontsize=8, fontweight='bold')
        
        # Remove o eixo original
        ax.axis('off')
        
        # # Imprime resumo no console
        # print("\n" + "="*70)
        # print("ESTATÍSTICAS DE REORDENAÇÃO DE ROTAS")
        # print("="*70)
        # for d_id, stat in zip(driver_ids, stats):
        #     print(f"\nMotorista {int(d_id)}:")
        #     print(f"  Total de reordenações: {stat['total_reorderings']}")
        #     print(f"  Bem-sucedidas: {stat['successful_reorderings']}")
        #     print(f"  Falhas: {stat['failed_reorderings']}")
        #     print(f"  Taxa de sucesso: {stat['success_rate']:.1f}%")
        #     print(f"  Tempo total economizado: {stat['total_time_saved']:.2f}")
        #     print(f"  Tempo total perdido: {stat['total_time_lost']:.2f}")
        #     print(f"  Impacto líquido em tempo: {stat['net_time_impact']:.2f}")
        #     print(f"  Distância total economizada: {stat['total_distance_saved']:.2f}")
        #     print(f"  Distância total aumentada: {stat['total_distance_increased']:.2f}")
        #     print(f"  Impacto líquido em distância: {stat['net_distance_impact']:.2f}")
        # print("="*70 + "\n")