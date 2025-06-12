import os
import numpy as np
from openpyxl import load_workbook
from shutil import copyfile

# Caminhos
TEMPLATE_PATH = '.\\objective_table\\template_objective_table.xlsx'
OUTPUT_PATH = 'objective_table.xlsx'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Volta da pasta /objective_table
ROOT_DIR = os.path.join(BASE_DIR, 'data', 'runs', 'teste')

# Mapear heurísticas para colunas por cenário
HEURISTIC_TO_COLUMN = {
    'initial_scenario': {
        'random_heuristic': 'D',
        'first_driver_heuristic': 'E',
        'nearest_driver_heuristic': 'F',
        'lowest_cost_driver_heuristic': 'G',
        'ppo_otimizado': 'H'
    },
    'medium_scenario': {
        'random_heuristic': 'J',
        'first_driver_heuristic': 'K',
        'nearest_driver_heuristic': 'L',
        'lowest_cost_driver_heuristic': 'M',
        'ppo_otimizado': 'N'
    },
    'complex_scenario': {
        'random_heuristic': 'P',
        'first_driver_heuristic': 'Q',
        'nearest_driver_heuristic': 'R',
        'lowest_cost_driver_heuristic': 'S',
        'ppo_otimizado': 'T'
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

copyfile(TEMPLATE_PATH, OUTPUT_PATH)
workbook = load_workbook(OUTPUT_PATH)

def preencher_planilhas(base_row, col, stats_dicts):
    for stat_key, sheet_name in STAT_KEY_TO_SHEET.items():
        stats = stats_dicts.get(stat_key)
        if stats is None:
            continue
        sheet = workbook[sheet_name]
        for i, metric in enumerate(METRIC_KEYS):
            cell = f'{col}{base_row + i}'
            sheet[cell] = stats.get(metric, 'N/A')

for obj_num in range(1, 11):
    obj_dir = os.path.join(ROOT_DIR, f'obj_{obj_num}')
    base_row = 3 + (obj_num - 1) * 4

    for scenario_name, heuristics_cols in HEURISTIC_TO_COLUMN.items():
        scenario_dir = os.path.join(obj_dir, scenario_name)

        for heuristic, col in heuristics_cols.items():
            npz_path = os.path.join(scenario_dir, heuristic, 'metrics_data.npz')

            if not os.path.exists(npz_path):
                print(f'Arquivo não encontrado: {npz_path}')
                continue

            try:
                data = np.load(npz_path, allow_pickle=True)

                stats_dicts = {}
                for key in STAT_KEY_TO_SHEET.keys():
                    arr = data.get(key)
                    stats_dicts[key] = arr.item() if isinstance(arr, np.ndarray) and arr.dtype == object else None

                preencher_planilhas(base_row, col, stats_dicts)

            except Exception as e:
                print(f'Erro ao processar {npz_path}: {e}')

# Salva o resultado
workbook.save(OUTPUT_PATH)
print(f"Arquivo salvo como {OUTPUT_PATH}")
