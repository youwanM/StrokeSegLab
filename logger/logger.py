from datetime import datetime
import logging
import os


def setup_logger(cli = True):
    logger=logging.getLogger()
    logger.setLevel(logging.INFO)

    if cli:
        handler = logging.StreamHandler()
    else:
        os.makedirs("logs",exist_ok=True)
        log_files = os.listdir("logs")
        print(len(log_files))
        if len(log_files) >= 5:
            log_files.sort() 
            os.remove(os.path.join("logs", log_files[0]))
        log_file=os.path.join("logs",f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)