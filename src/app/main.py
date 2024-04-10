# -*- coding: utf-8 -*-
"""
    Created by: Andrés Segura-Tinoco
    Version: 0.1.0
    Created on: Apr 10, 2024
    Created on: Apr 10, 2024
    Description: Main class of the argument mining framework.
"""

import gc
import json

def get_dict_from_json(json_path:str, encoding:str="utf-8") -> dict:
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

def start_app(app_setup:dict):
    print(app_setup)

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