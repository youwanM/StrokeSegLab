import logging
import os
from datetime import datetime

from utils.path import LOG_DIR

def setup_logger(cli: bool = True, verbose : bool = False) -> None:
    """
    Configure the global logger for the application:
    - If cli is True: configure a StreamHandler for logging into the console
    - If cli is False: create a logs directory (if it doesn't already exist),
      create a new log file inside it, and log both to file and console.
      Keeps only the 5 most recent files.
    - Message format: timestamp - level - message

    Args:
        cli (bool, optional): cli (True) or gui (False) mode. Defaults to True
        verbose (bool, optional): If True, sets the logging level to DEBUG. If False, sets it to INFO. Defaults to False.
    """
    logger = logging.getLogger()
    if verbose:
        logger.setLevel(logging.DEBUG)
    else : 
        logger.setLevel(logging.INFO)

    # Remove all existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    if cli:
        # Console only
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    else:
        # Create logs directory and manage files
        os.makedirs(LOG_DIR, exist_ok=True)
        log_files = sorted(os.listdir(LOG_DIR))
        if len(log_files) >= 5:
            os.remove(os.path.join(LOG_DIR, log_files[0]))
        
        log_file = os.path.join(LOG_DIR, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console handler (added even in GUI mode)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        
    logger.debug("Verbose mode enabled: logging level set to DEBUG.")
