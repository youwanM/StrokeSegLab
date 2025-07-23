import configparser
from utils.singleton import SingletonMeta
from utils.path import CONFIG_FILE

class Config(metaclass=SingletonMeta):
    def __init__(self):
        self.config_path = CONFIG_FILE
        self.config = configparser.ConfigParser()
        read_files = self.config.read(CONFIG_FILE)
        if not read_files:
            raise FileNotFoundError(f"Le fichier de config '{CONFIG_FILE}' est introuvable ou illisible.")

    
    def get(self, section: str, key: str):
        return self.config[section][key]
    
    def set(self, section: str, key: str, value: str):
        if section not in self.config:
            self.config.add_section(section)
        self.config[section][key] = value
        
    def save(self):
        with open(self.config_path, "w") as configfile:
            self.config.write(configfile)

    def clear(self,section):
        if self.config.has_section(section):
            self.config.remove_section(section)
        self.config.add_section(section)