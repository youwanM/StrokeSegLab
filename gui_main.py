import logging
import tkinter as tk
from tkinter import filedialog
import os

from inference.run_inference import Inference
from logger.logger import setup_logger
from manager.option_manager import Option
from postprocessing.postprocessor import Postprocessor
from preprocessing.preprocessor import Preprocessor
import torch

class GUIMain:
    def __init__(self):
        setup_logger(False)
        self.logger = logging.getLogger()
        self.option = Option()
        self.preprocessor = Preprocessor()
        self.inference = Inference()
        self.postprocessor = Postprocessor()

        self.option.set("device",self._check_device())
        self.option.set("output_path","./")


        self.window = tk.Tk()
        self.window.title('nnUNet prediction')
        self.input_path = tk.StringVar(value="")
        self.output_path = tk.StringVar(value="")
        self.frame = tk.Frame(self.window)
        self.frame.grid(row=0, column=0,padx=10,pady=10)

        self.nii_paths = []
        tk.Label(self.frame, text="Input path : ").grid(row=0,column=0,pady=10)
        self.label_input_path = tk.Label(self.frame,textvariable=self.input_path)
        self.label_input_path.grid(row=0,column=1)
        tk.Button(self.frame, text='Select input folder',command=self._select_input_folder).grid(row=0,column=2)
        tk.Button(self.frame, text='Select input file',command=self._select_input_file).grid(row=0,column=3)

        tk.Label(self.frame, text="Output path : ").grid(row=1,column=0,pady=10)
        self.label_output_path = tk.Label(self.frame,textvariable=self.output_path)
        self.label_output_path.grid(row=1,column=1)
        tk.Button(self.frame, text='Select output path',command=self._select_output_path).grid(row=1,column=2)

        tk.Button(self.frame, text='run', command=self._run).grid(row=2,column=1)

        self.window.mainloop()
    
    def _select_input_folder(self):
        self.input_path.set(filedialog.askdirectory(title='input path'))
        self.option.set("input_path",self.input_path.get())
        self._find_nii_files()
        
    def _select_output_path(self):
        self.output_path.set(filedialog.askdirectory(title='output path'))
        self.option.set('output_path',self.output_path.get())

    def _select_input_file(self):
        self.input_path.set(filedialog.askopenfilename(title='input path'))
        self.option.set("input_path",self.input_path.get())
        self._find_nii_files()

    def _find_nii_files(self):
        self.nii_paths = []
        input_path=self.option.get("input_path")
        if os.path.isfile(input_path) and input_path.endswith(".nii.gz"):
            self.nii_paths.append(input_path)
        elif os.path.isdir(input_path):
            for root, _, files in os.walk(input_path):
                for f in files:
                    if f.endswith(".nii.gz"):
                        self.nii_paths.append(os.path.join(root, f))
    
    def _run(self):
        for img in self.nii_paths:
            self.logger.info(f"Starting processing on: {img}")
            data, affine = self.preprocessor.run(img)
            data = self.inference.run(data)
            self.postprocessor.run(data,affine,img)
            self.preprocessor.clean()

    def _check_device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        self.logger.info(f"Using device: {device}")
        return device

if __name__ == "__main__":
  GUIMain()