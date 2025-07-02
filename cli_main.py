import argparse
from inference.run_inference import Inference
from manager.option_manager import Option
from postprocessing.postprocessor import Postprocessor
from preprocessing.preprocessor import Preprocessor
from logger.logger import setup_logger
import torch
import os
import logging
import tempfile

class CLIMain:
    def __init__(self, input_path, output_path = "./", model_path="./models/model.pth",patch_size=[128,128,128],suffix="seg"):
        setup_logger(True)
        self.logger = logging.getLogger()
        self.option = Option()
        self.option.set("input_path",input_path)
        self.option.set("output_path",output_path)
        self.option.set("model_path",model_path)
        self.option.set("suffix", suffix)
        self.option.set("device",self._check_device())
        self.preprocessor = Preprocessor()
        self.inference = Inference(patch_size=patch_size)
        self.postprocessor = Postprocessor()
    
    def _check_device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        self.logger.info(f"Using device: {device}")
        return device
    
    def _find_nii_files(self):
        nii_paths = []
        input_path=self.option.get("input_path")
        if os.path.isfile(input_path) and input_path.endswith((".nii.gz",".nii")) :
            nii_paths.append(input_path)
        elif os.path.isdir(input_path):
            for root, _, files in os.walk(input_path):
                for f in files:
                    if f.endswith((".nii.gz",".nii")):
                        nii_paths.append(os.path.join(root, f))
        self.logger.debug(f'nii_paths : {nii_paths}')
        return nii_paths
    
    def run(self):
        imgs=self._find_nii_files()
        temp_dir = tempfile.mkdtemp(prefix="unet_preprocess")
        for img in imgs:
            try:
                self.logger.info(f"Starting processing on: {img}")
                data, affine, bbox,original_shape, trsf_path, old_spacing, padding  = self.preprocessor.run(img,temp_dir)
                data = self.inference.run(data)
                self.postprocessor.run(data,affine,img,bbox,original_shape,temp_dir,trsf_path,old_spacing,padding)
            except Exception as e:
                self.logger.error(f"Erreur lors du traitement de {img} : {e}")
        self.preprocessor.clean(temp_dir)

def parse_args():
    parser = argparse.ArgumentParser(description="Run segmentation pipeline")
    parser.add_argument("-i", "--input", required=True, help="Input image path (required)")
    parser.add_argument("-o", "--output", help="Output folder (optional)")
    parser.add_argument("-m", "--model", help="Model path (optional)")
    parser.add_argument("-s", "--suffix", help="output name suffix (optional)")
    parser.add_argument("--patch_size", nargs=3, type=int, metavar=('X', 'Y', 'Z'), help="Patch size, 3 ints (optional)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    kwargs = {}
    if args.output:
        kwargs['output_path'] = args.output
    if args.model:
        kwargs['model_path'] = args.model
    if args.patch_size:
        kwargs['patch_size'] = args.patch_size
    if args.suffix:
        kwargs['suffix'] = args.suffix
    app = CLIMain(
        input_path=args.input,
        **kwargs
    )
    app.run()