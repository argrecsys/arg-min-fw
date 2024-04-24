# -*- coding: utf-8 -*-
"""
    Created by: Andrés Segura-Tinoco
    Version: 0.2.0
    Created on: Apr 10, 2024
    Updated on: Apr 24, 2024
    Description: Main class of the argument mining framework.
"""

import gc
import json
from framework.engine import Engine


def __get_dict_from_json(json_path: str, encoding: str = "utf-8") -> dict:
    result = {}
    try:
        with open(json_path, mode="r", encoding=encoding) as file:
            result = json.load(file)
    except Exception as e:
        print(e)
    return result


def get_dataset_metadata(dataset_name: str, task_number: int) -> tuple:

    if dataset_name == "dm-2019-annotated.csv":
        text_column = "sent_text"

        if task_number == 1:
            label_column = "sent_label1"
            label_dict = {"NO": 0, "YES": 1}

        elif task_number == 2:
            label_column = "sent_label2"
            label_dict = {"SPAM": 0, "CLAIM": 1, "PREMISE": 2}

    return text_column, label_column, label_dict


# Read data configuration
def read_fm_setup() -> dict:
    filepath = "src/app/settings/config.json"
    setup = __get_dict_from_json(filepath)
    return setup


# Use framework
def start_framework(fm_setup: dict):

    # Framework variables
    task_number = fm_setup["task"]
    dataset_name = fm_setup["dataset"]
    dataset_path = f"src/app/data/{dataset_name}"
    text_column, label_column, label_dict = get_dataset_metadata(
        dataset_name, task_number
    )

    # 1. Select the argument mining task of interest
    fm_engine = Engine(True, task_number)

    # 2. Select a tabular corpus and split it into 3 datasets for the selected task
    fm_engine.load_data(dataset_path, text_column, label_column, label_dict)
    fm_engine.split_data(
        test_size=0.1, validation_size=0.1, shuffled=True, random_state=42
    )
    # fm_engine.create_features()

    # 3. Choose learning algorithms/approaches:
    fm_engine.load_algorithms()


#####################
### START PROGRAM ###
#####################
if __name__ == "__main__":
    fm_setup = read_fm_setup()
    start_framework(fm_setup)
    gc.collect()
#####################
#### END PROGRAM ####
#####################
