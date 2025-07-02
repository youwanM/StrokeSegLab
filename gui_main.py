import logging
import tempfile
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
        self.preprocessor = Preprocessor(gui=self)
        self.inference = Inference(gui=self)
        self.postprocessor = Postprocessor(gui=self)

        self.option.set("device",self._check_device())
        self.option.set("output_path","./")

        self.window = tk.Tk()
        self.window.minsize(600, 150)
        self.window.title('nnUNet prediction')
        self.input_path = tk.StringVar(value="")
        self.output_path = tk.StringVar(value="")
        self.suffix = tk.StringVar(value="seg")

        self.status_text = tk.StringVar()
        self.working_on_text = tk.StringVar()
        self.result_text = tk.StringVar()

        self.frame = tk.Frame(self.window)
        self.frame.grid(row=0, column=0,padx=10,pady=10)
        

        self.nii_paths = []
        tk.Label(self.frame, text="Input path : ").grid(row=0,column=0,pady=10)
        self.entry_input_path = tk.Entry(self.frame, textvariable=self.input_path, state='readonly', width=50)
        self.entry_input_path.grid(row=0, column=1)
        tk.Button(self.frame, text='Select input folder',command=self._select_input_folder).grid(row=0,column=2)
        tk.Button(self.frame, text='Select input file',command=self._select_input_file).grid(row=0,column=3)

        tk.Label(self.frame, text="Output path : ").grid(row=1,column=0,pady=10)
        self.entry_output_path = tk.Entry(self.frame, textvariable=self.output_path, state='readonly', width=50)
        self.entry_output_path.grid(row=1, column=1)
        tk.Button(self.frame, text='Select output path',command=self._select_output_path).grid(row=1,column=2)

        tk.Label(self.frame, text="Suffix : ").grid(row=2, column=0, pady=10)
        self.entry_suffix = tk.Entry(self.frame, textvariable=self.suffix, width=20)
        self.entry_suffix.grid(row=2, column=1)

        tk.Button(self.frame, text='run', command=self._run).grid(row=3,column=1)

        self.label_working_on = tk.Label(self.frame, textvariable=self.working_on_text, fg="blue")
        self.label_status = tk.Label(self.frame, textvariable=self.status_text)
        self.label_result = tk.Label(self.frame, textvariable=self.result_text, font=("Arial", 14, "bold"))
        # self.stop_requested = False
        # self.stop_button = tk.Button(self.frame, text="Stop", command=self._stop)

        self.window.mainloop()
    
    def _select_input_folder(self):
        self.input_path.set(filedialog.askdirectory(title='input path'))
        self.option.set("input_path",self.input_path.get())
        
    def _select_output_path(self):
        self.output_path.set(filedialog.askdirectory(title='output path'))
        self.option.set('output_path',self.output_path.get())

    def _select_input_file(self):
        self.input_path.set(filedialog.askopenfilename(title='input path'))
        self.option.set("input_path",self.input_path.get())

    def _find_nii_files(self):
        self.nii_paths = []
        input_path=self.option.get("input_path")
        if os.path.isfile(input_path) and input_path.endswith((".nii.gz",".nii")):
            self.nii_paths.append(input_path)
        elif os.path.isdir(input_path):
            for root, _, files in os.walk(input_path):
                for f in files:
                    if f.endswith((".nii.gz",".nii")):
                        self.nii_paths.append(os.path.join(root, f))
        self.logger.debug(f'nii_paths : {self.nii_paths}')
    
    # def _stop(self):
    #     self.stop_requested = True
    #     self.result_text.set("🛑 Stopping...")
    #     self.label_result.config(fg="red")
    #     self.window.update()
    
    # def _check_stop(self):
    #     if self.stop_requested:
    #         self.logger.info("Prediction stopped by user.")
    #         self.succes = False
    #         return True

    def _run(self):
        self.success = True
        # self.stop_requested = False
        self._find_nii_files()
        self.option.set("suffix", self.suffix.get())
        self.label_working_on.grid(row=4, column=0, pady=10)
        self.label_status.grid(row=5, column=0, pady=10)
        self.label_result.grid(row=6, column=0, columnspan=4, pady=10)
        # self.stop_button.grid(row=3, column=2)
        self.result_text.set("⌛ Prediction running...")
        self.label_result.config(fg="blue")
        self.window.update()
        i=0
        len_nii_paths= len(self.nii_paths)
        temp_dir = tempfile.mkdtemp(prefix="unet_preprocess")
        for img in self.nii_paths:
            # if self._check_stop():
            #     break
            self.working_on_text.set(f"Working on: {os.path.basename(img)} ({i+1}/{len_nii_paths})")
            self.window.update()
            self.logger.info(f"Starting processing on: {img}")
            try:
                data, affine, bbox,original_shape, trsf_path, old_spacing, padding  = self.preprocessor.run(img,temp_dir)
                # if self._check_stop():
                #     break
                data = self.inference.run(data)
                # if self._check_stop():
                #     break
                self.postprocessor.run(data,affine,img,bbox,original_shape,temp_dir,trsf_path,old_spacing,padding)
                i+=1
            except Exception as e:
                self.logger.error(f"Erreur lors du traitement de {img} : {e}")
                self.success = False
                break
        self.preprocessor.clean(temp_dir)
        
        if self.success:
            self.result_text.set("✓ Success")
            self.label_result.config(fg="green")
            self.window.update()
        else:
            self.result_text.set("✗ Failed")
            self.label_result.config(fg="red")
            self.window.update()
    
    def update_status(self,s):
        self.status_text.set(s)
        self.window.update()



    def _check_device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        self.logger.info(f"Using device: {device}")
        return device

if __name__ == "__main__":
  GUIMain()