import configparser
from manager.singleton import SingletonMeta

class Config(metaclass=SingletonMeta):
    def __init__(self,config_path="./config/config.ini"):
        self.config_path = config_path
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
    
    def get(self, section: str, key: str):
        return self.config[section][key]
    
    def set(self, section: str, key: str, value: str):
        if section not in self.config:
            self.config.add_section(section)
        self.config[section][key] = value
        
    def save(self):
        with open(self.config_path, "w") as configfile:
            self.config.write(configfile)