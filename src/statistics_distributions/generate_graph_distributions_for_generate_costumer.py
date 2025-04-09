import numpy as np
import matplotlib.pyplot as plt
from src.main.utils.random_manager import RandomManager
from src.main.base.geometry import point_in_gauss_circle  # Importando sua função

# Parâmetros
centroid = (10, 20)  # Posição do Estabelecimento
radius = 20          # Raio de operação
limit = 50           # Grid Map Size
num_points = 1000    # Número de pontos gerados

# Criando gerador de números aleatórios
rng_generator = RandomManager().get_random_instance()

# Gerando pontos
points = [point_in_gauss_circle(centroid, radius, limit, rng_generator) for _ in range(num_points)]
x_vals, y_vals = zip(*points)  # Separando coordenadas X e Y

# Criando o gráfico
plt.figure(figsize=(6, 6))
plt.scatter(x_vals, y_vals, s=5, alpha=0.6, label="Pontos Gerados")
plt.scatter(*centroid, color="red", marker="x", label="Centro")
plt.xlim(0, limit)
plt.ylim(0, limit)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Distribuição da Geração de Clientes (pontos) - Gaussiano no Círculo")
plt.legend()
plt.grid(True)

# Salvando o gráfico
output_path = "/home/marcos/graficos/gauss_circle.png"
plt.savefig(output_path, dpi=300)
plt.show()

print(f"Gráfico salvo em {output_path}")
