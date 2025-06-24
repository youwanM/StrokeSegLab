import configparser
from manager.singleton import SingletonMeta

class Config(metaclass=SingletonMeta):
    def __init__(self,config_path="config.ini"):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
    
    def get(self, section: str, key: str):
        return self.config[section][key]