import yaml
import os

#read yaml file

def load_config():
  with open(os.path.join(os.path.dirname(__file__), 'config.yaml')) as file:
    config= yaml.safe_load(file)

  return config
