import logging
import shutil
import subprocess
import platform
from utils.config_manager import Config


class Viewer:
    def __init__(self):
        self.config = Config()
        self.logger = logging.getLogger()
        viewer = self.config.get('default','viewer')
        self.viewers = self.config.get('default', 'viewers').split(',')
        self.viewers = [v for v in self.viewers]
        if not viewer :
            self.update_path()
        else:
            path = self.config.get('ViewerPath',viewer)
            if not path:
                self.update_path()

    def update_path(self):
        exist = False
        for v in self.viewers:
            if v == "itksnap" and platform.system() == "Windows":
                    path = shutil.which("itk-snap")
            else:
                path = shutil.which(v)
            if path is not None :
                self.config.set('ViewerPath', v, path)
                self.config.set('default','viewer',v)
                exist = True
        if not exist:
            self.config.set('default','viewer',"")
        self.config.save()

    def check_viewer(self, viewer):
        viewers = self.config.get('default', 'viewers').split(',')
        viewers = [v for v in viewers]
        if viewer not in viewers:
            raise ValueError(f"Viewer '{viewer}' is not in the allowed list of viewers: {viewers}")
        
        path = self.config.get('ViewerPath', viewer)
        if not path:
            if viewer == "itksnap":
                if platform.system() == "Windows":
                    path = shutil.which("itk-snap")
                else :
                    shutil.which(viewer)
            else:
                path = shutil.which(viewer)
            if not path:
                raise FileNotFoundError(f"Viewer '{viewer}' not found in PATH.")

        self.config.set('ViewerPath', viewer, path)
        self.config.set('default', 'viewer', viewer)
        self.config.save()

    def run(self,img_path, seg_path):
        viewer = self.config.get("default","viewer")
        path = self.config.get("ViewerPath",viewer)
        if viewer ==  "itksnap":
            command = [path,"-g", img_path,"-s",seg_path]
        elif viewer == "fsleyes" :
            command = [path, img_path, seg_path]
        
        try:
            subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            self.update_path()
            self.logger.error(f"Failed to execute command {command}: {e}")

        