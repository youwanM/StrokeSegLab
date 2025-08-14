import logging
import shutil
import subprocess
import platform
from managers.config_manager import Config


class Viewer:
    """
    This class handle viewers : updating path, checking availability and running it
    """
    def __init__(self)-> None:
        """
        Initialize the viewer class 
        - Setting up config, logger and viewers (List of viewers supported by the application)
        - Check if the default viewer is available; if not, calling for an update.
        """
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

    def update_path(self) -> None:
        """
        Update all the path for the viewers supported by the application. Set a viewer available on the path as the default one
        """
        exist = False
        for v in self.viewers:
            if v == "itksnap" and platform.system() == "Windows":
                    path = shutil.which("itk-snap") # The correct shortcut on Windows is itk-snap, not itksnap
            else:
                path = shutil.which(v)
            if path is not None :
                self.config.set('ViewerPath', v, path)
                self.config.set('default','viewer',v)
                exist = True
        if not exist:
            self.config.set('default','viewer',"")
        self.config.save()

    def check_viewer(self, viewer : str)-> None:
        """
        Check if a viewer given is supported by the application and available on the path. If not raise an error based on the type of error. Only used in CLI mode, GUI users can simply select from a list of available viewers. If the viewer is available, set it as default one

        Args:
            viewer (str): name of the viewer
        """
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

    def run(self,img_path : str, seg_path : str)->None:
        """
        Open the base image and the generated segmentation in the default viewer. If an error occurs, call for an update of the viewers paths

        Args:
            img_path (str): Base image path (input)
            seg_path (str): Binary mask path (output)
        """
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

        