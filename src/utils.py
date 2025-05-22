import os
import yaml

def load_config(path=None):
    if path is None:
        # Caminho absoluto para config.yaml na pasta config na raiz do projeto
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.yaml')
    with open(path, 'r') as f:
        return yaml.safe_load(f)