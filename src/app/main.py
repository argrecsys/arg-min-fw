# -*- coding: utf-8 -*-
"""
    Created by: Andrés Segura-Tinoco
    Version: 0.2.0
    Created on: Apr 10, 2024
    Updated on: Feb 01, 2025
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
def read_framework_setup() -> dict:
    filepath = "src/app/settings/config.json"
    setup = __get_dict_from_json(filepath)
    return setup


# Use framework
def run_framework(fm_setup: dict):

    # 0. Read framework setup variables
    fm_setup = read_framework_setup()

    # Framework variables
    task_number = fm_setup["task"]
    dataset_name = fm_setup["dataset"]
    dataset_path = f"src/app/data/{dataset_name}"
    text_column, label_column, label_dict = get_dataset_metadata(
        dataset_name, task_number
    )

    # 1. Select the argument mining task of interest:
    # - T1: identification of argumentative fragments in an input text
    # - T2: classification of such fragments into argument components (e.g., claims and premises)
    # - T3: recognition of relations (e.g., support and attack) between pairs of argument components
    fm_engine = Engine(task_number, True)

    # 2. Select a tabular corpus and split it into 3 datasets for the selected task.
    # Tabular datasets should reflect the argument model used (commonly, in labels).
    fm_engine.load_data(dataset_path, text_column, label_column, label_dict)
    fm_engine.split_data(
        test_size=0.1, validation_size=0.1, shuffled=True, random_state=42
    )

    # 3. Choose learning algorithms/approaches:
    fm_engine.load_algorithms()

    # 4. Select hyperparameter optimization approach:
    # - Exhaustive search among discrete values with GridSearch (not recommended).
    # - Optimized search in uniform space with Optuna
    # - Optimized search in a logarithmic space with Optuna

    # 5. Select the metric(s) of interest: Accuracy, Precision, Recall, F1-score

    # 6. For all selected algorithms, run:
    # - Create model from training data
    # - Optimize hyperparameters with respect to the validation data
    # - Keep the configuration that minimizes the loss function
    # - Evaluate generated model with testing data
    # - Save results for the current model
    fm_engine.run_optimization(n_jobs=4, n_trials=50)


#####################
### START PROGRAM ###
#####################
if __name__ == "__main__":
    run_framework()
    gc.collect()
#####################
#### END PROGRAM ####
#####################
