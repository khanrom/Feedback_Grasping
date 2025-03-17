import yaml
import json
from types import SimpleNamespace
from pipelines.data_preprocess import DataPreProcessor

with open('config.yaml', 'r') as file:
    p = yaml.safe_load(file)
    params = json.loads(json.dumps(p), object_hook=lambda d: SimpleNamespace(**d))


if __name__ == "__main__":
    print("--Preprocess raw Jacquard Dataset--")
    data_processor = DataPreProcessor()
    data_processor.data2npy()
    print("--Done--")
