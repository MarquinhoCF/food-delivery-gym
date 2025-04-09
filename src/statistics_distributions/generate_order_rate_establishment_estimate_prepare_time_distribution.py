import numpy as np
import matplotlib.pyplot as plt
import os

# Diretório para salvar os gráficos
output_dir = "/home/marcos/graficos"
os.makedirs(output_dir, exist_ok=True)

# Parâmetros
order_production_time_rate = 40  # Tempo médio esperado (ajuste conforme necessário)
min_prepare_time = 20
max_prepare_time = 60
num_samples = 10000

# Função para gerar e salvar histogramas
def plot_and_save(samples, title, filename):
    plt.figure(figsize=(8, 5))
    plt.hist(samples, bins=40, density=True, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(min_prepare_time, color='red', linestyle='dashed', label=f'Min: {min_prepare_time}s')
    plt.axvline(max_prepare_time, color='green', linestyle='dashed', label=f'Max: {max_prepare_time}s')
    plt.title(title)
    plt.xlabel('Tempo de Preparação (s)')
    plt.ylabel('Densidade')
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Gráfico salvo em: {save_path}")



# 1️⃣ Distribuição Exponencial Truncada
samples = []
rng = np.random.default_rng()
while len(samples) < num_samples:
    time_to_prepare = round(rng.exponential(order_production_time_rate))
    if min_prepare_time <= time_to_prepare <= max_prepare_time:
        samples.append(time_to_prepare)
plot_and_save(samples, "Distribuição Exponencial Truncada", "exponencial_truncada.png")

# 2️⃣ Distribuição Normal (Gaussiana) com Clip
samples = np.random.normal(loc=order_production_time_rate, scale=7, size=num_samples)
clipped_samples = np.clip(np.round(samples), min_prepare_time, max_prepare_time)
plot_and_save(clipped_samples, "Distribuição Normal (Gaussiana)", "normal.png")

# 3️⃣ Distribuição Lognormal com Clip
samples = np.random.lognormal(mean=np.log(order_production_time_rate), sigma=0.2, size=num_samples)
clipped_samples = np.clip(np.round(samples), min_prepare_time, max_prepare_time)
plot_and_save(clipped_samples, "Distribuição Lognormal", "lognormal.png")

# 4️⃣ Distribuição Beta
samples = np.random.beta(a=2, b=5, size=num_samples)
samples = min_prepare_time + (max_prepare_time - min_prepare_time) * samples
plot_and_save(samples, "Distribuição Beta", "beta.png")
