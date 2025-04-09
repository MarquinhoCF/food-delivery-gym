import numpy as np
import matplotlib.pyplot as plt
import random

# Parâmetros
min_value = 10        # Valor mínimo gerado
max_value = 50        # Valor máximo gerado
num_samples = 10_000  # Quantidade de números gerados

# Gerando valores com randint
samples = [random.randint(min_value, max_value) for _ in range(num_samples)]

# Criando histograma
plt.figure(figsize=(8, 6))
plt.hist(samples, bins=range(min_value, max_value + 2), edgecolor='black', alpha=0.75, density=True, align='left')

# Configurações do gráfico
plt.xlabel("Valores gerados")
plt.ylabel("Frequência relativa")
plt.title(f"Distribuição da função randint [{min_value}, {max_value}]")
plt.xticks(range(min_value, max_value + 1))
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Salvando o gráfico
output_path = "/home/marcos/graficos/randint_distribution.png"
plt.savefig(output_path, dpi=300)
plt.show()

print(f"Gráfico salvo em {output_path}")
