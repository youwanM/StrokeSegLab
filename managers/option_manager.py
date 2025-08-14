from managers.singleton import SingletonMeta
from typing import TypeVar

T = TypeVar("T")

class Option(metaclass=SingletonMeta):
    """
    Singleton class to manage application option with a dictionnary.
    Options are execution parameters that change with each run (input path, save probability map, ...)

    Args:
        metaclass (_type_, optional): The metaclass controls the creation of the class itself. Defaults to SingletonMeta, which makes sure only one object of the class is ever created (singleton pattern)
    """
    def __init__(self)->None:
        """
        Initialize the option class with just a dictionnary
        """
        self._options = {}
    
    def get(self, key: str, default : T =None)->T:
        """
        Get the value for a key

        Args:
            key (str): Key name
            default (T, optional): Value to return if the key doesn't exist. Defaults to None

        Returns:
            T: Return value corresponding to the key
        """
        return self._options.get(key, default)
    
    def set(self, key: str, value :T)->None:
        """
        Set or add a value for a key

        Args:
            key (str): Key name
            value (T): Value corresponding to the key
        """
        self._options[key] = value