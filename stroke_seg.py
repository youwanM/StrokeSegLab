import argparse
import time
from gui.gui import GUIMain
from inference.inference import Inference
from utils.config_manager import Config
from utils.models_manager import add_model, update_models
from utils.option_manager import Option
from postprocessing.postprocessor import Postprocessor
from preprocessing.preprocessor import Preprocessor
from logger.logger import setup_logger
import onnxruntime as ort
import os
import logging
import tempfile
import sys

class CLIMain:
    def __init__(self, input_path,only_preprocessing, save_preprocessing,keep_MNI,save_pmap,threshold, model_name,suffix,viewer,import_model):
        setup_logger(True)
        self.logger = logging.getLogger()
        print("="*60)
        print("⚠️  This tool is for research purpose only ! ")
        print("="*60)
        self.option = Option()
        self.config = Config()
        if import_model is None or import_model == "__SHOW_MODELS__":
            self.option.set("input_path",input_path)
            self.only_preprocessing = only_preprocessing
            self.save_preprocessing = save_preprocessing 
            self.threshold = 0.5 if threshold is None else threshold
            if keep_MNI:
                self.option.set("keep_MNI", True)
            else:
                self.option.set("keep_MNI", False)
            if save_pmap:
                self.option.set("save_pmap", True)
            else:
                self.option.set("save_pmap",False)
            self.preprocessor = Preprocessor()
            if not self.only_preprocessing :
                if model_name !=None:
                    if os.path.isfile(model_name):
                        self.option.set("model_path",model_name)
                    else : 
                        models = self.config.get('default', 'models').split(',')
                        models = [m for m in models]
                        if model_name not in models:
                            self.logger.error(f"{model_name} not in {models}")
                            sys.exit(1)
                        else:
                            self.config.set("default","model",model_name)
                            self.config.save()
                if suffix != None:
                    self.config.set("default","suffix",suffix)
                self.viewer = viewer
                self.option.set("device",self._check_device())
                self.inference = Inference()
                self.postprocessor = Postprocessor()
        else:
            self.model_path = import_model
    
    def _check_device(self):    
        available_providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in available_providers:
            device = 'CUDAExecutionProvider'
        else : 
            device = 'CPUExecutionProvider'
        self.logger.info(f'using device : {device}')
        return device
    
    def import_model(self):
        try:
            update_models()
            model_name = add_model(self.model_path)
            self.logger.info(f"The model {model_name} was successfully imported")
            update_models()
        except ValueError as e:
            self.logger.error(f"Import failed: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error during model import: {e}")
    
    def show_models(self):
        update_models()
        models_str = self.config.get("default", "models")
        models = [m.strip() for m in models_str.split(',') if m.strip()]

        if not models:
            print("⚠️ No models found in the configuration.")
            return

        print("Available models:")
        for model in models:
            print(f" - {model}")

    def run(self):
        start = time.time()
        model_name = self.config.get("default","model")
        if self.config.get("ModelChannels",model_name)=="2":
            self.option.set("flair",True)
        else:
            self.option.set("flair",False)
        
        if not self.only_preprocessing and self.viewer != None and self.viewer != "default":
            try :
                self.postprocessor.check_viewer(self.viewer)
            except Exception as e:
                self.logger.error(f"Viewer check failed: {e}")
                raise
        if self.save_preprocessing:
            self.option.set("save_bet", True)
        else :
            self.option.set("save_bet", False)

        temp_dir = tempfile.mkdtemp(prefix="unet_preprocess")
        i=0
        nii_paths,subject,flair,none_list = self.preprocessor.find_nii_files()
        if len(nii_paths)==0:
            self.logger.error("No NIfTI files (.nii or .nii.gz) found in the specified input path.")
            raise ValueError("No NIfTI files found.")
        if self.option.get("flair") and subject != flair:
            if none_list:
                none_string = ", ".join(none_list)
                self.logger.info(f"These subjects : \"{none_string}\" are missing either a T1 or a FLAIR image.")
        self.logger.info(f"{len(nii_paths)} subject(s) found")
        for t1, flair in nii_paths.items():
            try:
                if flair is None:
                    self.logger.info(f"Starting processing on: {os.path.basename(t1)}")
                else : 
                    self.logger.info(f"Starting processing on: ({os.path.basename(t1)},{os.path.basename(flair)})")
                if not self.only_preprocessing :
                    data, affine, bbox,original_shape, trsf_path, old_spacing, padding,bet,MNI_base_image  = self.preprocessor.run(t1,flair,temp_dir)
                    data = self.inference.run(data)
                    if self.viewer != None and i==0:
                        self.postprocessor.run(data,affine,t1,bbox,original_shape,temp_dir,trsf_path,old_spacing,padding,bet,MNI_base_image,self.threshold,True)
                    else : 
                        self.postprocessor.run(data,affine,t1,bbox,original_shape,temp_dir,trsf_path,old_spacing,padding,bet,MNI_base_image,self.threshold)
                else : 
                    self.preprocessor.run(t1,flair,temp_dir,True)
                i+=1
            except Exception as e:
                self.logger.error(f"Error while processing {t1} : {e}")
        self.preprocessor.clean(temp_dir)
        final = time.time()-start
        self.logger.info(f"Total prediction time: {final:.2f} seconds")

def parse_args():
    parser = argparse.ArgumentParser(description="Run segmentation pipeline", epilog="If no arguments are provided, the graphical interface will be launched by default")
    parser.add_argument("-i", "--input", help="Input image(s) path")
    parser.add_argument("-m", "--model", help="Model name (optional)")
    parser.add_argument("-s", "--suffix", help="output name suffix (optional)")
    parser.add_argument("-V", "--viewer", nargs="?",const="default",help="Specify a viewer name, or use default if none is given")
    parser.add_argument("--only-preprocessing", action="store_true", help="Run only the preprocessing step and stop")
    parser.add_argument("--save-preprocessing", action="store_true", help="Save the brain-extracted image")
    parser.add_argument("--keep-mni", action="store_true", help="Save the input and output images registered to the MNI space")
    parser.add_argument("-t", "--threshold",  type=float, default=None, help="Threshold (optional)")
    parser.add_argument("--pmap", action="store_true", help="Save probability map")
    parser.add_argument("--import-model", nargs='?', const="__SHOW_MODELS__", help="Model path to add to usable models (try it with '-m', '--model' before). If called without value, list all models.")
    args = parser.parse_args()

    if (args.input and args.import_model) or (not args.input and not args.import_model):
        parser.error("You must specify either --input or --import-model")

    if args.import_model is not None and (args.save_preprocessing or args.keep_mni or args.pmap or args.model or args.only_preprocessing or args.suffix is not None or args.viewer is not None or args.threshold is not None):
        parser.error("When --import-model is used, no other options must be set!")

    if args.only_preprocessing and (args.save_preprocessing or args.keep_mni or args.pmap or args.model is not None or args.suffix is not None or args.viewer is not None or args.threshold is not None):
        parser.error("When --only-preprocessing is used, no other options must be set!")
    return args


if __name__ == "__main__":
    if len(sys.argv) == 1:
        GUIMain()
    else:
        args = parse_args()
        app = CLIMain(
            input_path=args.input,
            only_preprocessing=args.only_preprocessing,
            save_preprocessing=args.save_preprocessing,
            keep_MNI=args.keep_mni,
            save_pmap=args.pmap,
            model_name = args.model,
            threshold = args.threshold,
            suffix = args.suffix,
            viewer = args.viewer,
            import_model = args.import_model,
        )
        if args.import_model is None:
            app.run()
        elif args.import_model == "__SHOW_MODELS__":
            app.show_models()
        else:
            app.import_model()
                