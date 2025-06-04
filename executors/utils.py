import yaml
import sys
import logging


def load_yaml_config(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError as e:
        logging.error(f"File '{path}' not found.")
        sys.exit(-1)
