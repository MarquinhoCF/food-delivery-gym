import os
import numpy as np
from openpyxl import load_workbook
from shutil import copyfile
from openpyxl.styles import Font

# Caminhos
TEMPLATE_PATH = '/templates/template_objective_table.xlsx'
OUTPUT_PATH = 'objective_table.xlsx'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Volta da pasta /objective_table
ROOT_DIR = os.path.join(BASE_DIR, 'data', 'runs', 'execucoes')

# Mapear heurísticas para colunas por cenário
AGENT_TO_COLUMN = {
    'initial_scenario': {
        'random_heuristic': 'D',
        'first_driver_heuristic': 'E',
        'nearest_driver_heuristic': 'F',
        'lowest_cost_driver_heuristic': 'G',
        'ppo_otimizado_trained_18M_steps': 'H',
        'ppo_otimizado_trained_50M_steps': 'I',
        'ppo_otimizado_trained_100M_steps': 'J'
    },
    'medium_scenario': {
        'random_heuristic': 'L',
        'first_driver_heuristic': 'M',
        'nearest_driver_heuristic': 'N',
        'lowest_cost_driver_heuristic': 'O',
        'ppo_otimizado_trained_18M_steps': 'P',
        'ppo_otimizado_trained_50M_steps': 'Q',
        'ppo_otimizado_trained_100M_steps': 'R'
    },
    'complex_scenario': {
        'random_heuristic': 'T',
        'first_driver_heuristic': 'U',
        'nearest_driver_heuristic': 'V',
        'lowest_cost_driver_heuristic': 'W',
        'ppo_otimizado_trained_18M_steps': 'X',
        'ppo_otimizado_trained_50M_steps': 'Y',
        'ppo_otimizado_trained_100M_steps': 'Z'
    }
}

# Métricas por ordem
METRIC_KEYS = ['avg', 'std_dev', 'median', 'mode']

# Mapear nome do dicionário no .npz para aba no Excel
STAT_KEY_TO_SHEET = {
    'total_rewards_statistics': 'Recompensas',
    'time_spent_on_delivery_statistics': 'Tempo Efetivo Gasto',
    'total_distance_traveled_statistics': 'Distância Percorrida'
}

# Função auxiliar para aplicar negrito em um conjunto de células
def aplicar_negrito(sheet, col, base_row):
    bold_font = Font(bold=True)
    for i in range(4):  # média, desvio padrão, mediana, moda
        cell = sheet[f"{col}{base_row + i}"]
        cell.font = bold_font

# Função para destacar a melhor média de cada cenário
def destacar_melhor_media():
    print("Aplicando negrito nas melhores heurísticas por cenário...")
    for sheet_name in STAT_KEY_TO_SHEET.values():
        sheet = workbook[sheet_name]
        buscar_maior = sheet_name == "Recompensas"

        for obj_num in range(1, 11):
            base_row = 3 + (obj_num - 1) * 4

            for scenario_name, heuristics_cols in AGENT_TO_COLUMN.items():
                melhor_valor = None
                melhor_col = None

                for heuristic, col in heuristics_cols.items():
                    cell_value = sheet[f"{col}{base_row}"].value  # linha da média
                    try:
                        val = float(cell_value)
                        if melhor_valor is None:
                            melhor_valor = val
                            melhor_col = col
                        else:
                            if buscar_maior and val > melhor_valor:
                                melhor_valor = val
                                melhor_col = col
                            elif not buscar_maior and val < melhor_valor:
                                melhor_valor = val
                                melhor_col = col
                    except (TypeError, ValueError):
                        continue

                if melhor_col:
                    aplicar_negrito(sheet, melhor_col, base_row)

        print(f" - Concluído: {sheet_name}")

# Função para preencher as planilhas com os dados carregados
def preencher_planilhas(base_row, col, stats_dicts):
    for stat_key, sheet_name in STAT_KEY_TO_SHEET.items():
        stats = stats_dicts.get(stat_key)
        if stats is None:
            continue
        sheet = workbook[sheet_name]
        for i, metric in enumerate(METRIC_KEYS):
            cell = f'{col}{base_row + i}'
            sheet[cell] = stats.get(metric, 'N/A')

# ================== Execução do Script ====================

print("Copiando template para arquivo de saída...")
copyfile(BASE_DIR + TEMPLATE_PATH, OUTPUT_PATH)

print("Carregando workbook...")
workbook = load_workbook(OUTPUT_PATH)

print("Preenchendo planilhas com dados das execuções...")
for obj_num in range(1, 11):
    obj_dir = os.path.join(ROOT_DIR, f'obj_{obj_num}')
    base_row = 3 + (obj_num - 1) * 4
    print(f" - Processando objetivo {obj_num}...")

    for scenario_name, agents_cols in AGENT_TO_COLUMN.items():
        scenario_dir = os.path.join(obj_dir, scenario_name)
        print(f"   - Cenário: {scenario_name}")

        for agent, col in agents_cols.items():
            npz_path = os.path.join(scenario_dir, agent, 'metrics_data.npz')

            if not os.path.exists(npz_path):
                print(f"     [Aviso] Arquivo não encontrado: {npz_path}")
                continue

            try:
                data = np.load(npz_path, allow_pickle=True)

                stats_dicts = {}
                for key in STAT_KEY_TO_SHEET.keys():
                    arr = data.get(key)
                    stats_dicts[key] = arr.item() if isinstance(arr, np.ndarray) and arr.dtype == object else None

                preencher_planilhas(base_row, col, stats_dicts)

            except Exception as e:
                print(f"     [Erro] Falha ao processar {npz_path}: {e}")               

print("Concluído o preenchimento de dados. Destacando agentes...")
destacar_melhor_media()

print("Salvando o arquivo final...")
workbook.save(OUTPUT_PATH)

print(f"Processo concluído. Arquivo salvo como {OUTPUT_PATH}")
