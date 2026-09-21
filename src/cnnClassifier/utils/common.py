import json
import yaml
from pathlib import Path
from typing import Any
from box import ConfigBox
from box.exceptions import BoxValueError
from cnnClassifier import logger

def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads a YAML file and returns a dot-accessible ConfigBox."""
    try:
        with open(path_to_yaml, "r", encoding="utf-8") as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file loaded successfully from: {path_to_yaml}")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("YAML file is empty")
    except Exception as e:
        raise e

def create_directories(path_to_directories: list[Path | str], verbose: bool = True):
    """Creates a list of directories if they do not exist."""
    for path in path_to_directories:
        Path(path).mkdir(parents=True, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")

def save_json(path: Path, data: dict):
    """Saves dictionary data to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    logger.info(f"JSON file saved at: {path}")

def load_json(path: Path) -> ConfigBox:
    """Loads JSON file data into a ConfigBox."""
    with open(path, "r", encoding="utf-8") as f:
        content = json.load(f)
    logger.info(f"JSON file loaded successfully from: {path}")
    return ConfigBox(content)