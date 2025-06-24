from manager.singleton import SingletonMeta

class Option(metaclass=SingletonMeta):
    def __init__(self):
        self._options = {}
    
    def get(self, key: str, default=None):
        return self._options.get(key, default)
    
    def set(self, key: str, value):
        self._options[key] = value