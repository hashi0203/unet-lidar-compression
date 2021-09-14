import os
import pickle
import yaml
import config

def save_pickle(data, data_name, ext='.bin', path=config.data_path):
    with open(os.path.join(path, data_name+ext), 'wb') as f:
        pickle.dump(data, f)

def load_pickle(data_name, ext='.bin', path=config.data_path):
    with open(os.path.join(path, data_name+ext), 'rb') as f:
        data = pickle.load(f)
    return data

def load_yaml(yaml_name, ext='.yaml', path=config.data_path):
    with open(os.path.join(path, yaml_name+ext), 'r') as yml:
        calibration = yaml.load(yml, Loader=yaml.SafeLoader)
    return calibration
