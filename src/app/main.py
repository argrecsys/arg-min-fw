# -*- coding: utf-8 -*-
"""
    Created by: Andrés Segura-Tinoco
    Version: 0.1.0
    Created on: Apr 10, 2024
    Updated on: Apr 23, 2024
    Description: Main class of the argument mining framework.
"""

import gc
import json
from framework.engine import Engine


def get_dict_from_json(json_path: str, encoding: str = "utf-8") -> dict:
    result = {}
    try:
        with open(json_path, mode="r", encoding=encoding) as file:
            result = json.load(file)
    except Exception as e:
        print(e)
    return result


# Read data configuration
def read_app_setup() -> dict:
    filepath = "src/app/settings/config.json"
    setup = get_dict_from_json(filepath)
    return setup


def start_app(app_setup: dict):
    dateset_name = app_setup["dataset"]
    dataset_path = f"src/app/data/{dateset_name}"
    fm_engine = Engine()
    fm_engine.load_dataset(dataset_path)


#####################
### START PROGRAM ###
#####################
if __name__ == "__main__":
    app_setup = read_app_setup()
    start_app(app_setup)
    gc.collect()
#####################
#### END PROGRAM ####
#####################
