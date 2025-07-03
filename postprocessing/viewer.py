import configparser
import shutil
import subprocess
import os
import sys

from manager.config_manager import Config


class Viewer:
    def __init__(self):
        self.config = Config()
        self.viewer = self.config.get('default','viewer')
        self.path = self.config.get('ViewerPath',self.viewer)
        if not self.path:
            self.path = shutil.which(self.viewer)
            if self.path is not None:
                self.config.set('ViewerPath', self.viewer, self.path)
            else:
                viewers = self.config.get('default', 'viewers').split(',')
                viewers = [v for v in viewers if v and v != self.viewer]
                for v in viewers:
                    self.path = self.config.get('ViewerPath', v)
                    if self.path:
                        self.viewer = v
                        break
                    else:
                        self.path = shutil.which(v)
                        if self.path is not None:
                            self.config.set('ViewerPath', v, self.path)
                            self.viewer = v
                            break
        
        self.config.set('default', 'viewer', self.viewer)
        self.config.save()
    def check_viewer(self, viewer):
        viewers = self.config.get('default', 'viewers').split(',')
        viewers = [v for v in viewers]
        if viewer not in viewers:
            raise ValueError(f"Viewer '{viewer}' is not in the allowed list of viewers: {viewers}")
        
        path = self.config.get('ViewerPath', viewer)
        if not path:
            path = shutil.which(viewer)
            if not path:
                raise FileNotFoundError(f"Viewer '{viewer}' not found in PATH.")

        self.config.set('ViewerPath', viewer, path)
        self.viewer = viewer
        self.path = path
        self.config.set('default', 'viewer', self.viewer)
        self.config.save()

    def run(self,img_path, seg_path):
        if self.viewer == "itksnap":
            subprocess.Popen([self.path,"-g", img_path,"-s",seg_path])
        if self.viewer == "fsleyes" :
            subprocess.Popen([self.path, img_path, seg_path])

        
