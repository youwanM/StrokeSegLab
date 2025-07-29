import logging
import os
import subprocess
from utils.path import ANIMA_DIR

class AnimaWrapper:
    def __init__(self):
        self.logger = logging.getLogger()

    def run(self,command):
        exe = os.path.join(ANIMA_DIR,command[0])
        full_command = [exe] + command[1:]
        try:
            subprocess.run(
                full_command,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE
            )
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error while running executable {command[0]}: {e}")
            if e.stderr:
                self.logger.error(f"STDERR:\n{e.stderr.decode()}")
            raise