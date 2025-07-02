import logging
import os
import subprocess

class AnimaWrapper:
    def __init__(self,anima_path="./anima_scripts"):
        self.anima_path = anima_path
        self.logger = logging.getLogger()

    def run(self,command):
        scripts = os.path.join(self.anima_path,command[0])
        full_command = [scripts] + command[1:]
        try:
            result = subprocess.run(
                full_command,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE
            )
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error while running script {command[0]}: {e}")
            if e.stderr:
                self.logger.error(f"STDERR:\n{e.stderr.decode()}")
            raise