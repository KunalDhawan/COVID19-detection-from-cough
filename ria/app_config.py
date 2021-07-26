from environs import Env
import json
from typing import Dict, List
import os

curr_dir = os.path.dirname(os.path.realpath(__file__))

env = Env()
env.read_env()

def read_json_file(file:str) -> Dict:
    with open(file) as f:
        data = json.load(f)
    return data

ALLOWED_HOSTS: List[str] = [] #TODO: fixme
