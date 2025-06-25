import os
import subprocess

class AnimaWrapper:
    def __init__(self,anima_path="./anima_scripts"):
        self.anima_path = anima_path

    def run(self,command):
        scripts = os.path.join(self.anima_path,command[0])
        full_command = [scripts] + command[1:]
        subprocess.run(full_command,check=True)
