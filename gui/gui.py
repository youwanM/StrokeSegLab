import logging
import sys
import tempfile
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import os

from inference.run_inference import Inference
from logger.logger import setup_logger
from manager.config_manager import Config
from manager.option_manager import Option
from postprocessing.postprocessor import Postprocessor
from preprocessing.preprocessor import Preprocessor
import torch
import threading

class GUIMain:
    def __init__(self):
        setup_logger(False)
        self.logger = logging.getLogger()
        self.option = Option()
        self.config = Config()
        self.option.set("device",self._check_device())
        self.preprocessor = Preprocessor(gui=self)
        self.inference = Inference(gui=self)
        

        self.option.set("output_path","./")

        self.window = tk.Tk()
        self.window.minsize(800, 150)
        self.window.title('nnUNet prediction')
        self.input_path = tk.StringVar(value="")
        self.output_path = tk.StringVar(value="")
        self.suffix = tk.StringVar(value=self.config.get("default","suffix"))
        self.open_viewer = tk.BooleanVar()
        self.viewer = tk.StringVar(value=self.config.get('default','viewer'))
        self.viewer_error = tk.StringVar(value="")

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

        tk.Label(self.frame, text="Model : ").grid(row=3, column=0, pady=10)
        self.models = self.config.get("default","models").split(',')
        self.models = [m for m in self.models]
        self.combo_models = ttk.Combobox(self.frame,values=self.models)
        self.combo_models.current(self.models.index(self.config.get("default","model")))
        self.combo_models.grid(row =3, column =1)

        tk.Label(self.frame, text="Open viewer : ").grid(row=4, column=0, pady=10)
        viewer_button = tk.Checkbutton(self.frame, text="ON/OFF", variable=self.open_viewer)
        viewer_button.grid(row=4,column=1)
        entry_viewer = tk.Entry(self.frame, textvariable=self.viewer, width=10)
        entry_viewer.grid(row=4, column=2)
        viewer_label = tk.Label(self.frame, textvariable=self.viewer_error)
        viewer_label.grid(row=4, column=3, pady=10)
        viewer_label.config(fg="red")

        self.run_button = tk.Button(self.frame, text='run', command=self._run)
        self.run_button.grid(row=5,column=1)

        self.label_working_on = tk.Label(self.frame, textvariable=self.working_on_text, fg="blue")
        self.label_status = tk.Label(self.frame, textvariable=self.status_text)
        self.label_result = tk.Label(self.frame, textvariable=self.result_text, font=("Arial", 14, "bold"))
        self.stop_requested = False
        self.stop_button = tk.Button(self.frame, text="Stop", command=self._stop)

        self.window.mainloop()
    
    def _select_input_folder(self):
        self.input_path.set(filedialog.askdirectory(title='input path'))
        
        
    def _select_output_path(self):
        self.output_path.set(filedialog.askdirectory(title='output path'))

    def _select_input_file(self):
        self.input_path.set(filedialog.askopenfilename(title='input path'))

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
    
    def _stop(self):
        self.stop_requested = True
        self.result_text.set("🛑 Stopping...")
        self.label_result.config(fg="red")
        self.window.update()
    
    def _check_stop(self):
        if self.stop_requested:
            self.logger.info("Prediction stopped by user.")
            self.success = False
            return True
        

    def _run(self):
        self.run_button.config(state="disabled")
        self.viewer_error.set("")

        model_name = self.combo_models.get()
        if model_name not in self.models:
            self.logger.error(f"{model_name} not in {self.models}")
            sys.exit(1)
        else:
            self.config.set("default","model",model_name)
            self.config.save()
        self.option.set("open_viewer", self.open_viewer.get())
        self.postprocessor = Postprocessor(gui=self)

        self.option.set("input_path",self.input_path.get())
        self.option.set('output_path',self.output_path.get())

        self.logger.debug(f'viewer : {self.viewer.get()} open_viewer : {self.open_viewer.get()}')
        if self.open_viewer.get() and self.viewer.get() != self.config.get('default','viewer'):
            try :
                self.postprocessor.check_viewer(self.viewer.get())
            except Exception as e: 
                self.logger.error(f"Viewer check failed: {e}")
                self.viewer.set(self.config.get('default','viewer'))
                self.success = False
                self._update_result()
                self.viewer_error.set(e)
                raise
        
        self.success = True
        self.stop_requested = False
        self._find_nii_files()
        if self.config.get("default","suffix")!=self.suffix.get():
            self.config.set("default","suffix", self.suffix.get())
            self.config.save()
        self.label_working_on.grid(row=6, column=0, pady=10)
        self.label_status.grid(row=7, column=0, pady=10)
        self.label_result.grid(row=8, column=0, columnspan=4, pady=10)
        self.stop_button.grid(row=5, column=2)
        self.result_text.set("⌛ Prediction running...")
        self.label_result.config(fg="blue")
        t = threading.Thread(target=self._predict)
        t.start()



    def _predict(self):
        len_nii_paths= len(self.nii_paths)
        temp_dir = tempfile.mkdtemp(prefix="unet_preprocess")
        i=0
        for img in self.nii_paths:
            if self._check_stop():
                break
            s = f"Working on: {os.path.basename(img)} ({i+1}/{len_nii_paths})"
            self.window.after(0,self._update_stringvar,self.working_on_text,s)
            self.logger.info(f"Starting processing on: {img}")
            try:
                data, affine, bbox,original_shape, trsf_path, old_spacing, padding  = self.preprocessor.run(img,temp_dir)
                if self._check_stop():
                    break
                data = self.inference.run(data)
                if self._check_stop():
                    break
                if self.open_viewer.get() and i==0:
                    self.postprocessor.run(data,affine,img,bbox,original_shape,temp_dir,trsf_path,old_spacing,padding,True)
                else : 
                    self.postprocessor.run(data,affine,img,bbox,original_shape,temp_dir,trsf_path,old_spacing,padding)
                i+=1
            except Exception as e:
                self.logger.error(f"Erreur lors du traitement de {img} : {e}")
                self.success = False
                raise
        self.window.after(0,self._update_result)
        self.preprocessor.clean(temp_dir)

    def _update_result(self):
        self.label_working_on.grid_remove()
        self.label_status.grid_remove()
        if self.success:
            self.result_text.set("✓ Success")
            self.label_result.config(fg="green")
        else:
            if self.stop_requested:
                self.result_text.set("🛑 Stopped by user")
                self.label_result.config(fg="red")
            else :
                self.result_text.set("✗ Failed")
                self.label_result.config(fg="red")
        self.run_button.config(state="normal")
    def update_status(self,s):
        self.window.after(0,self._update_stringvar,self.status_text,s)

    def _update_stringvar(self,stringvar,s):
        stringvar.set(s)


    def _check_device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        self.logger.info(f"Using device: {device}")
        return device