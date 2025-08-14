import logging
import os
import subprocess
from utils.path import ANIMA_DIR

class AnimaWrapper:
    """
    Wrapper class to run Anima commands easily and safely
    """
    def __init__(self)->None:
        """
        Initialize the wrapper class
        """
        self.logger = logging.getLogger()

    def run(self,command:list[str])->None:
        """
        Run an Anima command, Raises error if the executable returns a non-zero exit status

        Args:
            command (list[str]): Command to run, where the first element is the executable name, and the rest are arguments
        """
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