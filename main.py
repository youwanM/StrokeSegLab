import argparse
from inference.run_inference import Inference
from manager.option_manager import Option
from postprocessing.postprocessor import Postprocessor
from preprocessing.preprocessor import Preprocessor
import torch
import os

class MainApp:
    def __init__(self, input_path, output_path = "./", model_path="./model.pth",patch_size=[128,128,128]):
        self.option = Option()
        self.option.set("input_path",input_path)
        self.option.set("output_path",output_path)
        self.option.set("model_path",model_path)
        self.option.set("device",self._check_device())
        self.preprocessor = Preprocessor()
        self.inference = Inference(patch_size)
        self.postprocessor = Postprocessor()
    
    def _check_device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print("Using device:", device)
        return device
    
    def _find_nii_files(self):
        nii_paths = []
        input_path=self.option.get("input_path")
        if os.path.isfile(input_path) and input_path.endswith(".nii.gz"):
            nii_paths.append(input_path)
        elif os.path.isdir(input_path):
            for root, _, files in os.walk(input_path):
                for f in files:
                    if f.endswith(".nii.gz"):
                        nii_paths.append(os.path.join(root, f))
        return nii_paths
    
    def run(self):
        imgs=self._find_nii_files()
        for img in imgs:
            print(f"Starting processing on: {img}")
            data, affine = self.preprocessor.run(img)
            data = self.inference.run(data)
            self.postprocessor.run(data,affine,img)

def parse_args():
    parser = argparse.ArgumentParser(description="Run segmentation pipeline")
    parser.add_argument("-i", "--input", required=True, help="Input image path (required)")
    parser.add_argument("-o", "--output", help="Output folder (optional)")
    parser.add_argument("-m", "--model", help="Model path (optional)")
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
    app = MainApp(
        input_path=args.input,
        **kwargs
    )
    app.run()