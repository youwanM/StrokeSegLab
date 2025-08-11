import argparse
from entrypoints.cli import CLIMain
from entrypoints.gui import GUIMain

import sys

from logger.logger import setup_logger
from utils.config_manager import Config

def restricted_float(x : str)->float:
    """
    Converts the input to a float and checks if it is between 0.0 and 1.0 (inclusive)
    This function is meant to be used with argparse to validate command-line arguments

    Args:
        x (str): The input value to convert and check.

    Returns:
        float: The valid float value between 0.0 and 1.0.
    """
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{x} is not a valid float")
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError(f"{x} is out of range: must be between 0 and 1")
    return x

def parse_args(gui : bool) -> argparse.Namespace:
    """
    Parse command-line arguments for the segmentation application
    - You must specify either --input or --import-model (not both)
    - If --import-model is used, no other option must be set
    - If --only-preprocessing is used, no other option must be set
    Args:
        gui (bool): True if no arguments were given to the script, meaning it runs in GUI mode.
    Returns:
        argparse.Namespace: Parsed command-line arguments as attributes.
    """
    parser = argparse.ArgumentParser(description="Run segmentation pipeline", epilog="If no arguments are provided, the graphical interface will be launched by default")
    parser.add_argument("-i", "--input", help="Input image(s) path")
    parser.add_argument("-m", "--model", help="Model name (optional)")
    parser.add_argument("-s", "--suffix", help="output name suffix (optional)")
    parser.add_argument("-V", "--viewer", nargs="?",const="default",help="Specify a viewer name, or use default if none is given")
    parser.add_argument("--only-preproc", action="store_true", help="Run only the preprocessing step and stop")
    parser.add_argument("--save-preproc", action="store_true", help="Save all the preprocessing steps")
    parser.add_argument("--keep-mni", action="store_true", help="Save the input and output images registered to the MNI space")
    parser.add_argument("-t", "--threshold",  type=restricted_float, default=None, help="Threshold (optional)")
    parser.add_argument("--pmap", action="store_true", help="Save probability map")
    parser.add_argument("--import-model", nargs='?', const="__SHOW_MODELS__", help="Model path to add to usable models (try it with '-m', '--model' before). If called without value, list all models.")
    parser.add_argument("--gui", action="store_true", help="Open the GUI application")
    parser.add_argument("-v", "--verbose",action="store_true", help="Enable verbose mode (set logging level to DEBUG instead of INFO).")
    args = parser.parse_args()
    if args.gui or gui:
        if args.import_model :
            parser.error("You can't open the GUI in import model mode")
    else :
        if (args.input and args.import_model) or (not args.input and not args.import_model):
            parser.error("You must specify either --input or --import-model")

        if args.import_model is not None and (args.keep_mni or args.pmap or args.model or args.only_preproc or args.suffix is not None or args.viewer is not None or args.threshold is not None or args.save_preproc):
            parser.error("When --import-model is used, no other options must be set!")

    if args.only_preproc and (args.keep_mni or args.pmap or args.model is not None or args.suffix is not None or args.viewer is not None or args.threshold is not None or args.save_preproc):
        parser.error("When --only-preprocessing is used, no other options must be set!")
    return args


if __name__ == "__main__":
    config = Config()
    gui = len(sys.argv) == 1
    args = parse_args(gui)
    if gui or args.gui: # No command-line arguments or gui argument provided: launch the graphical interface
        setup_logger(cli=False,verbose=args.verbose)
        GUIMain(input_path=args.input,
                only_preprocessing=args.only_preproc,
                keep_MNI=args.keep_mni,
                save_pmap=args.pmap,
                model_name = args.model,
                threshold = args.threshold,
                suffix = args.suffix,
                viewer = args.viewer,
                save_preprocessing = args.save_preproc,
                )
    else:
        # Initialize the CLI application with parsed arguments
        setup_logger(verbose=args.verbose)
        app = CLIMain(
            input_path=args.input,
            only_preprocessing=args.only_preproc,
            keep_MNI=args.keep_mni,
            save_pmap=args.pmap,
            model_name = args.model,
            threshold = args.threshold,
            suffix = args.suffix,
            viewer = args.viewer,
            import_model = args.import_model,
            save_preprocessing = args.save_preproc,
        )
        if args.import_model is None:
            app.run() # Prediction or brain extraction mode
        elif args.import_model == "__SHOW_MODELS__":
            app.show_models() # Show model mode
        else:
            app.import_model() # Import model mode
                