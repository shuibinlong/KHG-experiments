import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_json_config(config_path):
    logging.info(" Loading configuration ".center(100, "-"))
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config