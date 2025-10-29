import os
import sys

from utils.string import APP_NAME

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if sys.platform == "win32":
    ROAMING = os.getenv("APPDATA")
    LOCAL = os.getenv("LOCALAPPDATA")
    PROGRAMDATA = os.getenv("PROGRAMDATA")
    ATLAS_DIR = os.path.join(PROGRAMDATA, APP_NAME, "Atlas")
    MODEL_DIR = os.path.join(PROGRAMDATA, APP_NAME, "Models")
elif sys.platform == "darwin":
    # macOS
    ROAMING = os.path.expanduser("~/Library/Application Support")
    LOCAL = ROAMING
else:
    # Linux
    ROAMING = os.path.expanduser("~/.config")
    LOCAL = os.path.expanduser("~/.local/share")
    ATLAS_DIR = os.path.join(BASE_DIR, "Atlas")
    MODEL_DIR = os.path.join(BASE_DIR, APP_NAME, "Models")
    
CONFIG_FILE = os.path.join(ROAMING, APP_NAME, "config.ini")
ANIMA_DIR = os.path.join(BASE_DIR, "../Anima")
LOGO_INRIA = os.path.join(BASE_DIR, "assets", "INRIA.png")
LOGO_INSTITUTIONS = os.path.join(BASE_DIR, "assets", "LOGO_INSTITUTIONS.png")
LOGO = os.path.join(BASE_DIR, "assets", "brain.png")
LOG_DIR = os.path.join(LOCAL, APP_NAME,"logs")
USER_GUIDE = os.path.join(BASE_DIR, "USER_GUIDE.md")
