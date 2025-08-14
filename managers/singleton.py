class SingletonMeta(type):
    """
    Metaclass that ensures only one instance of a class exists (Singleton pattern)
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Each time a class using this metaclass is instantiated, this method is called. It checks if an instance of the class already exists in the _instances dictionary:
        - If no instance exists, it creates a new one by calling the superclass __call__, stores it in _instances, and returns it
        - If an instance already exists, it returns the stored instance instead of creating a new one
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]