import yaml
import os

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'config.yaml')
    token_list_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'token_list.yaml')
    
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    with open(token_list_path, 'r') as token_file:
        token_list = yaml.safe_load(token_file)
    
    config['token_addresses'] = token_list['tokens']
    
    return config