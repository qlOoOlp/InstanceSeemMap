import json
import os
import yaml
from utils.utils import load_config
from .metrics.evaluation import evaluation

def append_results_to_json(output_path, data):
    with open(output_path, 'w') as f:
        json.dump([data], f, indent=4)

def __main__():
    config = load_config("config/evaluate.yaml")
    eval = evaluation(config)
    results = eval.evaluate()
    output_path = os.path.join(config["data_path"],"evals",f"evaluation_results_{config['vlm']}.json")
    append_results_to_json(output_path, results)

if __name__ == "__main__":
    __main__()