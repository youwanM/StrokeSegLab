import time
from inference.inference import Inference
from managers.config_manager import Config
from utils.models_manager import add_model, get_input_channels, update_models
from managers.option_manager import Option
from postprocessing.postprocessor import Postprocessor
from preprocessing.preprocessor import Preprocessor
import onnxruntime as ort
import os
import logging
import tempfile

class CLIMain:
    """
    Command line tool for the segmentation application
    """
    def __init__(self, input_path : str,only_preprocessing : bool, save_preprocessing : bool, keep_MNI : bool ,save_pmap : bool ,skip_BET : bool, threshold : float , model_name : str ,suffix : str ,viewer : str,import_model : str)->None:
        """
        Initialize the CLI.

        **Args:**
        - `input_path` (str): The input path.
        - `only_preprocessing` (bool): Perform brain extraction only.
        - `save_preprocessing` (bool): Save all preprocessing steps.
        - `keep_MNI` (bool): Save input and segmentation in MNI space.
        - `save_pmap` (bool): Save probability map in addition to binary mask.
        - `skip_BET` (bool): Skip brain extraction step.
        - `threshold` (float): Segmentation threshold (default 0.5 if None).
        - `model_name` (str): Path or name of the model.
        - `suffix` (str): Output segmentation suffix.
        - `viewer` (str): Viewer name for result visualization.
        - `import_model` (str): Model import mode or name.
        """
        self._logger = logging.getLogger()
        print("="*60)
        print("⚠️  This tool is for research purpose only ! ")
        print("="*60)
        self._option = Option()
        self._config = Config()
        if import_model is None : 
            self._option.set("input_path",input_path)
            self.only_preprocessing = only_preprocessing
            self.threshold = 0.5 if threshold is None else threshold
            self._option.set("save_preproc", save_preprocessing)
            self._option.set("keep_MNI", keep_MNI)
            self._option.set("save_pmap", save_pmap)
            self._option.set("skip_BET", skip_BET)
            self.preprocessor = Preprocessor()
            if not self.only_preprocessing :
                if model_name !=None:
                    if os.path.isfile(model_name):
                        self._option.set("model_path",model_name)
                    else : 
                        models = self._config.get('default', 'models').split(',')
                        models = [m for m in models]
                        if model_name not in models: # Check if the model specified is in the config models list 
                            self._logger.error(f"{model_name} not in {models}")
                            sys.exit(1)
                        else:
                            self._config.set("default","model",model_name)
                            self._config.save()
                if suffix != None:
                    self._config.set("default","suffix",suffix)
                self.viewer = viewer
                self._option.set("device",self._check_device())
                self.inference = Inference()
                self.postprocessor = Postprocessor()

        elif import_model == "__SHOW_MODELS__":
            pass
        else:
            self.model_path = import_model
    
    def _check_device(self) -> str:
        """
        @public
        Check Cuda if available

        Returns:
            str: The selected execution provider ('CUDAExecutionProvider' or 'CPUExecutionProvider')
        """
        available_providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in available_providers:
            device = 'CUDAExecutionProvider'
        else : 
            device = 'CPUExecutionProvider'
        self._logger.info(f'using device : {device}')
        return device
    
    def import_model(self) -> None:
        """
        Try to import the model file in the models directory
        """
        try:
            update_models()
            model_name = add_model(self.model_path)
            self._logger.info(f"The model {model_name} was successfully imported")
            update_models()
        except ValueError as e:
            self._logger.error(f"Import failed: {e}")
        except Exception as e:
            self._logger.error(f"Unexpected error during model import: {e}")
    
    def show_models(self) -> None:
        """
        Print all the models available in the models directory
        """
        update_models()
        models_str = self._config.get("default", "models")
        models = [m.strip() for m in models_str.split(',') if m.strip()]

        if not models:
            print("⚠️ No models found in the configuration.")
            return

        print("Available models:")
        for model in models:
            print(f" - {model}")

    def run(self)-> None:
        """
        Run the prediction or the brain extraction only with all the options specified
        """
        start = time.time()
        if self._option.get("model_path") is None:
            model_name = self._config.get("default","model")
            if self._config.get("ModelChannels",model_name)=="2":
                self._option.set("flair",True)
            else:
                self._option.set("flair",False)
        else :
            channels = get_input_channels(self._option.get("model_path"))
            if channels==2:
                self._option.set("flair",True)
            else:
                self._option.set("flair",False)

        
        if not self.only_preprocessing and self.viewer != None and self.viewer != "default":
            try :
                self.postprocessor.check_viewer(self.viewer)
            except Exception as e:
                self._logger.error(f"Viewer check failed: {e}")
                raise

        
        i=0
        nii_paths,subject,flair,none_list = self.preprocessor.find_nii_files()
        if len(nii_paths)==0:
            self._logger.error("No NIfTI files (.nii or .nii.gz) found in the specified input path.")
            raise ValueError("No NIfTI files found.")
        if self._option.get("flair") and subject != flair:
            if none_list:
                none_string = ", ".join(none_list)
                self._logger.info(f"These subjects : \"{none_string}\" are missing either a T1 or a FLAIR image.")
        self._logger.info(f"{len(nii_paths)} subject(s) found")
        for t1, flair in nii_paths.items():
            temp_dir = tempfile.mkdtemp(prefix="unet_preprocess")
            try:
                if flair is None:
                    self._logger.info(f"Starting processing on: {os.path.basename(t1)}")
                else : 
                    self._logger.info(f"Starting processing on: ({os.path.basename(t1)},{os.path.basename(flair)})")
                if not self.only_preprocessing :
                    data, affine, bbox,original_shape, trsf_path, old_spacing, padding,bet,MNI_base_image  = self.preprocessor.run(t1,flair,temp_dir)
                    data = self.inference.run(data)
                    if self.viewer != None and i==0: # Only open the viewer for the first image
                        self.postprocessor.run(data,affine,t1,bbox,original_shape,temp_dir,trsf_path,old_spacing,padding,bet,MNI_base_image,self.threshold,True)
                    else : 
                        self.postprocessor.run(data,affine,t1,bbox,original_shape,temp_dir,trsf_path,old_spacing,padding,bet,MNI_base_image,self.threshold)
                else : # Preprocessing only
                    self.preprocessor.run(t1,flair,temp_dir,True)
                i+=1
                self.preprocessor.clean(temp_dir)
            except Exception as e: # If an exception occurs, processing of the current subject is stopped, a message is logged to inform the user, and the process continues with the next subject
                self._logger.error(f"Error while processing {t1} : {e}")
        self.preprocessor.clean(temp_dir)
        final = time.time()-start
        self._logger.info(f"Total prediction time: {final:.2f} seconds")