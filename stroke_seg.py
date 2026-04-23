import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import argparse
from entrypoints.cli import CLIMain
from entrypoints.gui import GUIMain

from utils.logger import setup_logger
from managers.config_manager import Config

# ... [Rest of your stroke_seg.py code] ...

def restricted_float(x: str) -> float:
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{x} is not a valid float")
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError(f"{x} is out of range: must be between 0 and 1")
    return x

# REMOVE the 'gui' parameter here
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stroke Segmentation Frontier Edition")
    
    # Frontier Placeholders
    parser.add_argument("-session", "--session_dir", help="Path to {SuspendDirectory}")
    parser.add_argument("-input", "--input_dir", help="Path to {NonDicomAppInputDirectory}")
    parser.add_argument("-result", "--result_dir", help="Path to {NonDicomAppResultDirectory}")
    
    # Mode Flags
    parser.add_argument("--gui", action="store_true", help="Launch Configuration UI")
    parser.add_argument("--console", action="store_true", help="Launch Segmentation Engine")
    
    # Parameters
    parser.add_argument("-t", "--threshold", type=restricted_float, default=0.5)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-m", "--model", help="Model name")
    parser.add_argument("--skip-bet", action="store_true")

    return parser.parse_args()

if __name__ == "__main__":
    config = Config()
    
    # Call without arguments now
    args = parse_args() 
    
    if args.gui:
        setup_logger(cli=False, verbose=args.verbose)
        # Ensure your GUIMain.__init__ only takes (self, args)
        app = GUIMain(args) 
        app.run_frontier_config() 

    elif args.console:
        setup_logger(verbose=args.verbose)
        
        # Load threshold from the session directory if available
        if args.session_dir:
            thresh_file = os.path.join(args.session_dir, "threshold.txt")
            if os.path.exists(thresh_file):
                with open(thresh_file, "r") as f:
                    args.threshold = float(f.read().strip())
        
        app = CLIMain(args)
        app.run()