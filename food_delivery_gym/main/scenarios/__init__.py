from importlib.resources import files

def get_all_scenarios() -> list[str]:
    """Descobre todo os cenários disponíveis varrendo o diretório de scenarios."""
    scenarios_path = files("food_delivery_gym.main.scenarios")
    return sorted(
        p.stem
        for p in scenarios_path.iterdir()
        if p.is_file() and p.suffix == ".json"
    )

def get_defaults_scenarios() -> list[str]:
    """Retorna os cenários default."""
    return ["initial", "medium", "complex"]