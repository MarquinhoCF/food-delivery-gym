import numpy as np
import matplotlib.pyplot as plt
import random

# Parâmetros
min_value = 3       # Valor mínimo gerado
max_value = 5       # Valor máximo gerado
num_samples = 10_000  # Quantidade de números gerados

# Gerando valores com uniform
samples = [random.uniform(min_value, max_value) for _ in range(num_samples)]

# Criando histograma
plt.figure(figsize=(8, 6))
plt.hist(samples, bins=50, edgecolor='black', alpha=0.75, density=True)

# Configurações do gráfico
plt.xlabel("Valores gerados")
plt.ylabel("Densidade de frequência")
plt.title(f"Distribuição da função uniform [{min_value}, {max_value}]")
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Salvando o gráfico
output_path = "/home/marcos/graficos/uniform_distribution.png"
plt.savefig(output_path, dpi=300)
plt.show()

print(f"Gráfico salvo em {output_path}")
