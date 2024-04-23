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

    # Task variables
    dateset_name = app_setup["dataset"]
    dataset_path = f"src/app/data/{dateset_name}"
    text_column = "sent_text"
    label_column = "sent_label1"
    label_dict = {"NO": 0, "YES": 1}
    num_labels = len(label_dict)

    # Use framework
    fm_engine = Engine(True)
    fm_engine.load_data(dataset_path, text_column, label_column, label_dict)
    fm_engine.split_data(
        test_size=0.1, validation_size=0.1, shuffled=True, random_state=42
    )
    fm_engine.create_features()


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
