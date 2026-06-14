from pathlib import Path
import yaml


CONFIG_PATH = Path(__file__).parent / "config.yml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def get_paddleocr_config() -> dict:
    return load_config()["paddleocr"]