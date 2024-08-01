from functools import lru_cache
from pathlib import Path

import yaml


@lru_cache(maxsize=1)
def load_config(yaml_path: Path) -> dict:
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


config = load_config(Path('config.yaml'))
