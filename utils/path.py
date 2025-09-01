import os
import sys

from utils.string import APP_NAME

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if sys.platform == "win32":
    ROAMING = os.getenv("APPDATA")
    LOCAL = os.getenv("LOCALAPPDATA")
    PROGRAMDATA = os.getenv("PROGRAMDATA")
elif sys.platform == "darwin":
    # macOS
    ROAMING = os.path.expanduser("~/Library/Application Support")
    LOCAL = ROAMING
else:
    # Linux
    ROAMING = os.path.expanduser("~/.config")
    LOCAL = os.path.expanduser("~/.local/share")
    
CONFIG_FILE = os.path.join(ROAMING, APP_NAME, "config.ini")
ANIMA_DIR = os.path.join(BASE_DIR, "anima")
ATLAS_DIR = os.path.join(PROGRAMDATA, APP_NAME, "Atlas")
MODEL_DIR = os.path.join(PROGRAMDATA, APP_NAME, "Models")
LOGO_INRIA = os.path.join(BASE_DIR,"StrokeSegCore", "assets", "INRIA.png")
LOGO = os.path.join(BASE_DIR,"StrokeSegCore", "assets", "brain.png")
LOG_DIR = os.path.join(LOCAL,APP_NAME,"logs")
USER_GUIDE = os.path.join(BASE_DIR,"StrokeSegCore","USER_GUIDE.md")
