from datetime import datetime
import logging
import os


def setup_logger(cli :bool = True)->None:
    """
    Configure the global logger for the application :
    - If cli is True : configure a StreamHandler for logging into the console
    - If cli is False : Create a logs directory if it doesn’t already exist, then create a new log file inside it. Keeps only the 5 most recent files. Configure a FileHandler for logginh into the log file created
    - Mesage format : timestamp - level - message
    Args:
        cli (bool, optional): cli (True) or gui (False) mode. Defaults to True
    """
    logger=logging.getLogger() # Get the root logger instance (singleton)
    logger.setLevel(logging.DEBUG) # Set the logging level to DEBUG, which means the logger will capture all messages with level DEBUG and higher (all the message)

    if cli:
        handler = logging.StreamHandler()
    else:
        os.makedirs("logs",exist_ok=True)
        log_files = os.listdir("logs")
        if len(log_files) >= 5:
            log_files.sort() 
            os.remove(os.path.join("logs", log_files[0]))
        log_file=os.path.join("logs",f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log") # Example filename: 'logs/20250729_153045.log'Ω
        handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)