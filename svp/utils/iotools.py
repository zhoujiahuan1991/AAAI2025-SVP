import os
import yaml
import json
from typing import Union

def save_train_configs(path, args):
    if args.if_no_logging==False:
        if not os.path.exists(path):
            os.makedirs(path)
        with open(f'{path}/configs.yaml', 'w') as f:
            yaml.dump(vars(args), f, default_flow_style=False)


def read_json(filename: str) -> Union[list, dict]:
    """read json files"""
    with open(filename, "r", encoding="utf-8") as fin:
        data = json.load(fin)
    return data