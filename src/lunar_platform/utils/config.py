import yaml
import os
from pathlib import Path

def get_config(config_name: str) -> dict:
    """Loads a configuration YAML file from the configs/ directory."""
    config_path = Path(f"configs/{config_name}.yaml")
    if not config_path.exists():
        # fallback for tests if run from different dir
        config_path = Path(__file__).parent.parent.parent.parent / f"configs/{config_name}.yaml"
        if not config_path.exists():
             return {}
             
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config.get(config_name, config)
