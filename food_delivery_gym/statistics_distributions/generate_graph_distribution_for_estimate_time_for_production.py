import numpy as np
import matplotlib.pyplot as plt
import os

# Diretório para salvar os gráficos
output_dir = "/home/marcos/graficos"
os.makedirs(output_dir, exist_ok=True)

# Parâmetros
order_production_time_rate = 60  # Tempo médio esperado
min_prepare_time = 20
max_prepare_time = 60
num_samples = 10000

# Função para gerar e salvar histogramas
def plot_and_save(samples, title, filename):
    plt.figure(figsize=(8, 5))
    plt.hist(samples, bins=40, density=True, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(min_prepare_time, color='red', linestyle='dashed', label=f'Min: {min_prepare_time}s')
    plt.axvline(max_prepare_time, color='green', linestyle='dashed', label=f'Max: {max_prepare_time}s')
    plt.axvline(order_production_time_rate, color='orange', linestyle='dashed', label=f'Média esperada: {order_production_time_rate}s')
    plt.title(title)
    plt.xlabel('Tempo de Preparação (s)')
    plt.ylabel('Densidade')
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Gráfico salvo em: {save_path}")

# 1️⃣ Distribuição Beta Ajustada ao Tempo Médio
relative_position = (order_production_time_rate - min_prepare_time) / (max_prepare_time - min_prepare_time)
a = 1 + 5 * relative_position
b = 7 - a

samples = np.random.beta(a, b, size=num_samples)
samples = min_prepare_time + (max_prepare_time - min_prepare_time) * samples

plot_and_save(samples, "Distribuição Beta Ajustada", "beta_ajustada.png")
