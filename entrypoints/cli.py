import time
import os
import logging
import sys
import onnxruntime as ort
from inference.inference import Inference
from managers.config_manager import Config
from utils.models_manager import add_model, get_input_channels, update_models
from managers.option_manager import Option
from postprocessing.postprocessor import Postprocessor
from preprocessing.preprocessor import Preprocessor

class CLIMain:
    """
    Command line tool for the segmentation application adapted for syngo.via Frontier.
    """
    def __init__(self, args) -> None:
        self._logger = logging.getLogger()
        print("="*60)
        print("⚠️  This tool is for research purpose only! ")
        print("="*60)
        
        self.args = args
        self._option = Option()
        self._config = Config()
        
        # 1. Handle syngo.via Directories
        self.result_dir = args.result_dir 
        self.session_dir = args.session_dir # This is {SuspendDirectory}
        
        # Mapping for components (Preprocessor/Postprocessor)
        self._option.set("input_path", args.input_dir)
        self._option.set("result_path", self.result_dir)
        self._option.set("session_path", self.session_dir)

        # 2. Handle Model Import Logic
        if hasattr(args, 'import_model') and args.import_model is not None:
            self.model_path = args.import_model
            return

        # 3. Configure Pipeline Options
        self.only_preprocessing = getattr(args, 'only_preproc', False)
        self.threshold = args.threshold
        
        self._option.set("save_preproc", getattr(args, 'save_preproc', False))
        self._option.set("keep_MNI", getattr(args, 'keep_mni', False))
        self._option.set("save_pmap", getattr(args, 'pmap', False))
        self._option.set("skip_BET", getattr(args, 'skip_bet', False))
        
        self.preprocessor = Preprocessor()

        if not self.only_preprocessing:
            self._setup_inference_engine(args)

    def _setup_inference_engine(self, args):
        model_name = getattr(args, 'model', None)
        if model_name:
            if os.path.isfile(model_name):
                self._option.set("model_path", model_name)
            else:
                models = self._config.get('default', 'models').split(',')
                if model_name not in [m.strip() for m in models]:
                    self._logger.error(f"{model_name} not found in available models.")
                    sys.exit(1)
                else:
                    self._config.set("default", "model", model_name)
                    self._config.save()

        if getattr(args, 'suffix', None):
            self._config.set("default", "suffix", args.suffix)
            
        self.viewer = getattr(args, 'viewer', None)
        self._option.set("device", self._check_device())
        self.inference = Inference()
        self.postprocessor = Postprocessor()

    def _check_device(self) -> str:
        available_providers = ort.get_available_providers()
        device = 'CUDAExecutionProvider' if 'CUDAExecutionProvider' in available_providers else 'CPUExecutionProvider'
        self._logger.info(f'Using device: {device}')
        return device

    def run(self) -> None:
        """
        Run prediction using the session_dir as the working directory.
        """
        start = time.time()
        
        # 1. Determine the base working directory
        # We use session_dir if provided, otherwise fallback to local temp
        base_work_dir = self.session_dir if self.session_dir else os.getcwd()
        
        # Configure Channels
        model_name = self._config.get("default", "model")
        is_flair = self._config.get("ModelChannels", model_name) == "2"
        self._option.set("flair", is_flair)

        # Find Files
        nii_paths, subject, flair, none_list = self.preprocessor.find_nii_files()
        if not nii_paths:
            self._logger.error("No NIfTI files found.")
            return

        self._logger.info(f"{len(nii_paths)} subject(s) found. Using working dir: {base_work_dir}")

        for i, (t1, flair_img) in enumerate(nii_paths.items()):
            # Create a subject-specific subfolder in the session directory 
            # to avoid file collisions between subjects.
            subj_name = os.path.basename(t1).split('.')[0]
            temp_dir = os.path.join(base_work_dir, f"work_{subj_name}")
            os.makedirs(temp_dir, exist_ok=True)
            
            try:
                self._logger.info(f"Processing: {os.path.basename(t1)}")
                
                if not self.only_preprocessing:
                    # 1. Preprocessing
                    data, affine, bbox, orig_shape, trsf, spacing, pad, bet, mni = \
                        self.preprocessor.run(t1, flair_img, temp_dir)
                    
                    # 2. Inference
                    data = self.inference.run(data)
                    
                    # 3. Postprocessing (Saves final mask to result_dir)
                    show_viewer = (self.viewer is not None and i == 0)
                    self.postprocessor.run(
                        data, affine, t1, bbox, orig_shape, temp_dir,
                        trsf, spacing, pad, bet, mni, self.threshold, 
                        show_viewer,
                        output_dir=self.result_dir
                    )
                else:
                    self.preprocessor.run(t1, flair_img, temp_dir, True)
                
            except Exception as e:
                self._logger.error(f"Error processing {t1}: {e}")
            finally:
                # Clean up the subject sub-folder, but keep the session_dir root 
                # (where threshold.txt lives).
                self.preprocessor.clean(temp_dir)

        final = time.time() - start
        self._logger.info(f"Total prediction time: {final:.2f} seconds")