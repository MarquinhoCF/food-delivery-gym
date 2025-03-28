import numpy as np
import matplotlib.pyplot as plt

# Definição dos parâmetros
tempo = np.linspace(0, 10000, 500)  # Intervalo de tempo na simulação

# Função sigmoide para penalização: cresce lentamente no início e acelera após 1400
def penalizacao(t, t_shift=(576/10)*25, k=0.005):
    return 5 * (0.02 / (0.02 + np.exp(-k * (t - t_shift))))  # Sigmoide deslocada

# Calculando os valores da penalização
penalizacoes = penalizacao(tempo)

# Plotando o gráfico
plt.figure(figsize=(8, 5))
plt.plot(tempo, penalizacoes, label="Penalização Sigmoide", color='blue', linewidth=2)
plt.xlabel("Tempo na Simulação")
plt.ylabel("Valor da Penalização")
plt.title("Progressão da Penalização Sigmoide pelo Tempo")
plt.axvline(x=1400, color='red', linestyle='--', label="Aceleração da Penalização")
plt.legend()
plt.grid(True)

# Exibir o gráfico
plt.show()
