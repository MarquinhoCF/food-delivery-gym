from __future__ import annotations

import argparse
import os
import sys
import traceback

from food_delivery_gym.main.statistic.simulation_stats import (
    npz_to_json,
    json_to_npz,
)

def convert_npz_to_json(npz_path: str) -> str:
    """Converte NPZ → JSON no mesmo diretório. Retorna o caminho de saída."""
    out_path = os.path.splitext(npz_path)[0] + ".json"
    npz_to_json(npz_path, out_path)
    return out_path


def convert_json_to_npz(json_path: str) -> str:
    """Converte JSON → NPZ no mesmo diretório. Retorna o caminho de saída."""
    out_path = os.path.splitext(json_path)[0] + ".npz"
    json_to_npz(json_path, out_path)
    return out_path


def find_metrics_files(directory: str, base_name: str) -> tuple[list[str], list[str]]:
    """
    Vasculha `directory` recursivamente e retorna dois grupos:
      - npz_files: arquivos <base_name>.npz encontrados
      - json_files: arquivos <base_name>.json encontrados
    """
    npz_files:  list[str] = []
    json_files: list[str] = []

    for root, _dirs, files in os.walk(directory):
        for fname in files:
            if fname == f"{base_name}.npz":
                npz_files.append(os.path.join(root, fname))
            elif fname == f"{base_name}.json":
                json_files.append(os.path.join(root, fname))

    return npz_files, json_files

def _do_convert(src: str, converter, label_in: str, label_out: str, dry_run: bool) -> bool:
    """Executa (ou simula) uma conversão individual. Retorna True se ok."""
    base_out = os.path.splitext(src)[0] + f".{label_out}"
    print(f"  {label_in.upper()} → {label_out.upper()}  {src}")
    print(f"             ↳  {base_out}")
    if dry_run:
        return True
    try:
        out = converter(src)
        print(f"  ✓ Salvo em: {out}")
        return True
    except Exception as exc:
        print(f"  ✗ Erro: {exc}")
        traceback.print_exc()
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Converte métricas SimulationStats entre NPZ e JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "path",
        help="Arquivo (.npz ou .json) ou diretório a processar.",
    )
    parser.add_argument(
        "--name",
        default="metrics_data",
        metavar="NOME",
        help="Nome-base para busca em diretório (padrão: metrics_data).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mostra o que seria convertido sem gravar nada.",
    )
    args = parser.parse_args()

    path    = args.path
    dry_run = args.dry_run

    if dry_run:
        print("[DRY-RUN] Nenhum arquivo será gravado.\n")

    if os.path.isfile(path):
        ext = os.path.splitext(path)[1].lower()
        if ext == ".npz":
            ok = _do_convert(path, convert_npz_to_json, "npz", "json", dry_run)
        elif ext == ".json":
            ok = _do_convert(path, convert_json_to_npz, "json", "npz", dry_run)
        else:
            print(f"Extensão não reconhecida: '{ext}'. Use .npz ou .json.")
            sys.exit(1)
        sys.exit(0 if ok else 1)

    if os.path.isdir(path):
        npz_files, json_files = find_metrics_files(path, args.name)
        total = len(npz_files) + len(json_files)

        if total == 0:
            print(f"Nenhum arquivo '{args.name}.npz' ou '{args.name}.json' "
                  f"encontrado em '{path}'.")
            sys.exit(0)

        print(f"Encontrados: {len(npz_files)} NPZ + {len(json_files)} JSON "
              f"(base: '{args.name}')\n")

        errors = 0
        for f in npz_files:
            ok = _do_convert(f, convert_npz_to_json, "npz", "json", dry_run)
            if not ok:
                errors += 1
            print()

        for f in json_files:
            ok = _do_convert(f, convert_json_to_npz, "json", "npz", dry_run)
            if not ok:
                errors += 1
            print()

        print(f"Concluído: {total - errors}/{total} conversões bem-sucedidas.")
        sys.exit(0 if errors == 0 else 1)

    print(f"Caminho não encontrado: '{path}'")
    sys.exit(1)


if __name__ == "__main__":
    main()