import argparse
import time
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
    def __init__(self, input_path, output_path = "./", model_path="./models/model_fp16.pth",patch_size=[128,128,128],suffix="seg",viewer=None):
        setup_logger(True)
        self.logger = logging.getLogger()
        self.option = Option()
        self.option.set("input_path",input_path)
        self.option.set("output_path",output_path)
        self.option.set("model_path",model_path)
        self.option.set("suffix", suffix)
        if viewer != None:
            self.option.set("open_viewer",True)
        else :
            self.option.set("open_viewer",False)
        self.viewer = viewer
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
        self.logger.debug(f'Nombre de fichiers nii trouvés : {len(nii_paths)}')
        return nii_paths
    
    def run(self):
        start = time.time()
        imgs=self._find_nii_files()
        if not imgs:
            self.logger.error("No NIfTI files (.nii or .nii.gz) found in the specified input path.")
            raise
        
        if self.viewer != None and self.viewer != "default":
            try :
                self.postprocessor.check_viewer(self.viewer)
            except Exception as e:
                self.logger.error(f"Viewer check failed: {e}")
                raise

        temp_dir = tempfile.mkdtemp(prefix="unet_preprocess")
        i=0
        for img in imgs:
            try:
                self.logger.info(f"Starting processing on: {img}")
                data, affine, bbox,original_shape, trsf_path, old_spacing, padding  = self.preprocessor.run(img,temp_dir)
                data = self.inference.run(data)
                if self.viewer != None and i==0:
                    self.postprocessor.run(data,affine,img,bbox,original_shape,temp_dir,trsf_path,old_spacing,padding,True)
                else : 
                    self.postprocessor.run(data,affine,img,bbox,original_shape,temp_dir,trsf_path,old_spacing,padding)
                i+=1
            except Exception as e:
                self.logger.error(f"Error while processing {img} : {e}")
                self.preprocessor.clean(temp_dir)
                raise
        self.preprocessor.clean(temp_dir)
        final = time.time()-start
        self.logger.info(f"Total prediction time: {final:.2f} seconds")

def parse_args():
    parser = argparse.ArgumentParser(description="Run segmentation pipeline")
    parser.add_argument("-i", "--input", required=True, help="Input image path (required)")
    parser.add_argument("-o", "--output", help="Output folder (optional)")
    parser.add_argument("-m", "--model", help="Model path (optional)")
    parser.add_argument("-s", "--suffix", help="output name suffix (optional)")
    parser.add_argument("--patch_size", nargs=3, type=int, metavar=('X', 'Y', 'Z'), help="Patch size, 3 ints (optional)")
    parser.add_argument("-V", "--viewer", nargs="?",const="default",help="Specify a viewer name, or use default if none is given")
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
    if args.viewer:
        kwargs['viewer'] = args.viewer
    app = CLIMain(
        input_path=args.input,
        **kwargs
    )
    app.run()