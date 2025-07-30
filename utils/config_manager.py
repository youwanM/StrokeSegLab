import configparser
from utils.singleton import SingletonMeta
from utils.path import CONFIG_FILE

class Config(metaclass=SingletonMeta):
    """
    Singleton class to manage application configuration with an INI files
    The config stores default parameters that are shared across runs and depend on the machine setup (viewer path, model names, output suffix, ...)

    Args:
        metaclass (_type_, optional): The metaclass controls the creation of the class itself. Defaults to SingletonMeta, which makes sure only one object of the class is ever created (singleton pattern)
    """

    def __init__(self)->None:
        """
        Initialize the Config singleton by reading the config file
        Raises a FileNotFoundError if the config file does not exist or cannot be read
        """
        self.config_path = CONFIG_FILE
        self.config = configparser.ConfigParser()
        read_files = self.config.read(CONFIG_FILE)
        if not read_files:
            raise FileNotFoundError(f"Le fichier de config '{CONFIG_FILE}' est introuvable ou illisible.")

    
    def get(self, section: str, key: str)->str:
        """
        Get the value for a key in a section

        Args:
            section (str): Section name in the config file
            key (str): Key name in the config file

        Returns:
            str: Value corresponding to the key in the section
        """
        return self.config[section][key]
    
    def set(self, section: str, key: str, value: str)-> None:
        """
        Set or add a value for a key in a given section
        If the section does not exist, it will be created

        Args:
            section (str): Section name in the config file
            key (str): Key name in the config file
            value (str): Value to assign
        """
        if section not in self.config:
            self.config.add_section(section)
        self.config[section][key] = value
        
    def save(self)-> None:
        """
        Save the current configuration back to the config file
        """
        with open(self.config_path, "w") as configfile:
            self.config.write(configfile)

    def clear(self,section:str)->None:
        """
        Remove all keys from a section by removing and re-adding the section

        Args:
            section (str): Section name in the config file
        """
        if self.config.has_section(section):
            self.config.remove_section(section)
        self.config.add_section(section)