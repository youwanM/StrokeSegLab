import logging
import sys
import tempfile
import tkinter as tk
from tkinter import Menu, filedialog, ttk, messagebox
import onnxruntime as ort
import os

from gui.string import APP_NAME, DEVELOPERS, HELP, LICENSE, PUBLICATIONS, VERSION
from inference.inference import Inference
from logger.logger import setup_logger
from manager.config_manager import Config
from manager.option_manager import Option
from manager.path import LOGO
from postprocessing.postprocessor import Postprocessor
from preprocessing.preprocessor import Preprocessor
import threading

class GUIMain:
    def __init__(self):
        setup_logger(False)
        self.logger = logging.getLogger()
        self.logger.warning("="*60)
        self.logger.warning("This tool is for research purpose only ! ")
        self.logger.warning("="*60)
        self.option = Option()
        self.config = Config()
        self.option.set("device",self._check_device())
        self.preprocessor = Preprocessor(gui=self)
        self.inference = Inference(gui=self)
        self.postprocessor = Postprocessor(gui=self)
        self.running = False
        self.nii_paths = {}

        self.window = tk.Tk()
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)
        self.window.title(APP_NAME)
        self.input_path = tk.StringVar(value="")
        self.output_path = tk.StringVar(value="")
        self.suffix = tk.StringVar(value=self.config.get("default","suffix"))
        self.open_viewer = tk.BooleanVar()
        self.save_bet = tk.BooleanVar()

        self.channel_text = tk.StringVar()
        self.status_text = tk.StringVar()
        self.working_on_text = tk.StringVar()
        self.result_text = tk.StringVar()

        menubar = Menu(self.window)
        help_menu = Menu(menubar, tearoff=0)
        help_menu.add_command(label='Help', command=self._show_help)
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label='Help',menu=help_menu)
        self.window.config(menu=menubar)

        
        frame = tk.Frame(self.window)
        frame.grid()
        tk.Label(frame, text="Input path : ").grid(row=0,column=0,pady=10)
        self.entry_input_path = tk.Entry(frame, textvariable=self.input_path, state='readonly', width=50)
        self.entry_input_path.grid(row=0, column=1)
        tk.Button(frame, text='Select input folder',command=self._select_input_folder).grid(row=0,column=2)
        tk.Button(frame, text='Select input file',command=self._select_input_file).grid(row=0,column=3)

        tk.Label(frame, text="Output path : ").grid(row=1,column=0,pady=10)
        self.entry_output_path = tk.Entry(frame, textvariable=self.output_path, state='readonly', width=50)
        self.entry_output_path.grid(row=1, column=1)
        tk.Button(frame, text='Select output path',command=self._select_output_path).grid(row=1,column=2)

        self.label_suffix = tk.Label(frame, text="Suffix : ")
        self.entry_suffix = tk.Entry(frame, textvariable=self.suffix, width=20)

        self.label_model = tk.Label(frame, text="Model : ")
        self.models = self.config.get("default","models").split(',')
        self.models = [m for m in self.models]
        self.label_channel = tk.Label(frame,textvariable=self.channel_text,fg="blue")
        self.combo_models = ttk.Combobox(frame,values=self.models,state="readonly")
        self.combo_models.bind("<<ComboboxSelected>>", self._on_model_change)
        self.combo_models.current(self.models.index(self.config.get("default","model")))
        self._on_model_change()

        self.label_open_viewer = tk.Label(frame, text="Open viewer : ")
        self.viewer_button = tk.Checkbutton(frame, text="ON/OFF", variable=self.open_viewer)
        v = self.config.get("default", "viewers").split(",")
        self.viewers = [v for v in v if self.config.get("ViewerPath", v)]
        if len(self.viewers)==0:
            self.viewer_button.config(state="disabled")
            self.label_viewer_not_found = tk.Label(frame, text="No viewer found",fg="red")
        else :
            self.combo_viewers = ttk.Combobox(frame,values=self.viewers,state="readonly")
            self.combo_viewers.current(self.viewers.index(self.config.get("default","viewer")))
        
        self.label_save_bet = tk.Label(frame,text="Save brain-extracted image : ")
        self.save_bet_button = tk.Checkbutton(frame, text="YES/NO", variable=self.save_bet)

        execution_modes = ["Prediction", "Brain extraction only"]
        self.label_exec_mode = tk.Label(frame, text="Execution mode : ")
        self.label_exec_mode.grid(row=6, column=0)
        self.combo_modes = ttk.Combobox(frame,values=execution_modes,state="readonly")
        self.combo_modes.current(0)
        self.combo_modes.grid(row=6, column=1)
        self.combo_modes.bind("<<ComboboxSelected>>", self._on_mode_change)
        self.run_button = tk.Button(frame, text='run', command=self._run, state="disabled")
        self.run_button.grid(row=6,column=2)
        self._on_mode_change()

        self.label_working_on = tk.Label(frame, textvariable=self.working_on_text, fg="blue")
        self.label_status = tk.Label(frame, textvariable=self.status_text)
        self.label_result = tk.Label(frame, textvariable=self.result_text, font=("Arial", 14, "bold"))
        self.stop_requested = False
        self.stop_button = tk.Button(frame, text="Stop", command=self._stop)


        messagebox.showwarning(title='Research Purpose Only', message='This tool is for research purpose only !')
        self.window.mainloop()

    def _check_paths_filled(self):
        if self.input_path.get() and self.output_path.get():
            self.run_button.config(state='normal')
        else:
            self.run_button.config(state='disabled') 

    def _on_model_change(self,event=None):
        model = self.combo_models.get()
        channels = self.config.get("ModelChannels",model)
        if channels == "2":
            self.channel_text.set("Using FLAIR and T1")
        else:
            self.channel_text.set("Using T1")

    def _on_mode_change(self,event=None):
        mode = self.combo_modes.get()
        if mode == "Prediction":
            self.label_suffix.grid(row=2, column=0, pady=10)
            self.entry_suffix.grid(row=2, column=1)
            self.label_model.grid(row=3, column=0, pady=10)
            self.combo_models.grid(row =3, column =1)
            self.label_open_viewer.grid(row=4, column=0, pady=10)
            self.viewer_button.grid(row=4,column=1)
            self.label_save_bet.grid(row=5,column=0)
            self.save_bet_button.grid(row=5,column=1)
            self.label_channel.grid(row=3,column=2)
            if len(self.viewers)==0:
                self.label_viewer_not_found.grid(row=4, column=2)
            else : 
                self.combo_viewers.grid(row=4, column=2)
        else : 
            self.label_suffix.grid_remove()
            self.entry_suffix.grid_remove()
            self.label_model.grid_remove()
            self.combo_models.grid_remove()
            self.label_open_viewer.grid_remove()
            self.viewer_button.grid_remove()
            self.label_save_bet.grid_remove()
            self.save_bet_button.grid_remove()
            self.label_channel.grid_remove()
            if len(self.viewers)==0:
                self.label_viewer_not_found.grid_remove()
            else : 
                self.combo_viewers.grid_remove()
            

    def _show_about(self):
        size = 500
        about_window= tk.Toplevel(self.window)
        tk.Label(about_window, text=APP_NAME, font=("Arial", 18, "bold")).pack(pady=10)
        tk.Label(about_window,text=VERSION).pack(pady=10)
        about_window.logo = tk.PhotoImage(file=LOGO)
        tk.Label(about_window,image=about_window.logo).pack()
        about_notebook = ttk.Notebook(about_window)
        about_notebook.pack(expand=True, fill='both', padx=10, pady=10)
        about_window.title("About")
        tk.Button(about_window, text="Close", command=about_window.destroy).pack(anchor='e',padx=10,pady=[0,10])

        developers_frame = tk.Frame(about_notebook)
        about_notebook.add(developers_frame, text='Developers')
        tk.Label(developers_frame, text=DEVELOPERS,justify="left",wraplength=size).pack(anchor='w')

        license_frame = tk.Frame(about_notebook)
        about_notebook.add(license_frame, text = 'License')
        tk.Label(license_frame, text = LICENSE, justify='left',wraplength=size ).pack(anchor='w')

        publications_frame = tk.Frame(about_notebook)
        about_notebook.add(publications_frame, text = 'Publications')
        for title, citation in PUBLICATIONS:
            tk.Label(publications_frame, text=title, font=("Arial", 12, "bold"), justify='left', wraplength=size).pack(anchor='w', pady=(5, 0))
            tk.Label(publications_frame, text=citation, justify='left', wraplength=size).pack(anchor='w', pady=(0, 5))

    def _show_help(self):
        size =500
        help_window= tk.Toplevel(self.window)
        tk.Label(help_window, text=APP_NAME, font=("Arial", 18, "bold")).pack(pady=10)
        tk.Label(help_window,text=VERSION).pack(pady=10)
        help_window.logo = tk.PhotoImage(file=LOGO)
        tk.Label(help_window, text = HELP, justify='left',wraplength=size ).pack(anchor='w')
        tk.Button(help_window, text="Close", command=help_window.destroy).pack(anchor='e',padx=10,pady=[0,10])

    def _select_input_folder(self):
        self.input_path.set(filedialog.askdirectory(title='input path'))
        self._check_paths_filled()
        
    def _select_output_path(self):
        self.output_path.set(filedialog.askdirectory(title='output path'))
        self._check_paths_filled()

    def _select_input_file(self):
        self.input_path.set(filedialog.askopenfilename(title='input path'))
        self._check_paths_filled()

    def _find_nii_files(self):
        path_list = []
        input_path=self.option.get("input_path")
        if os.path.isfile(input_path) and input_path.endswith((".nii.gz",".nii")):
            self.nii_paths[input_path]=None
        elif os.path.isdir(input_path):
            for root, _, files in os.walk(input_path):
                for f in files:
                    if f.endswith((".nii.gz",".nii")):
                        if self.option.get("flair"):
                            path_list.append(os.path.join(root, f))
                        else:
                            self.nii_paths[os.path.join(root, f)]=None
            if self.option.get("flair"):
                subject = {}
                for f in path_list:
                    name = os.path.basename(f)
                    if "_T1" in name:
                        subject_id = f.split("_T1")[0]
                        subject.setdefault(subject_id,{})['T1']=f
                    elif "_FLAIR" in name:
                        subject_id = f.split("_FLAIR")[0]
                        subject.setdefault(subject_id,{})['FLAIR']=f
                for subject_id,modalities in subject.items():
                    if "T1" in modalities:
                        self.nii_paths[modalities["T1"]]=modalities.get('FLAIR',None)
        self.logger.debug(f'nii_paths : {self.nii_paths}')

    def _on_close(self):
        if self.running:
            messagebox.showwarning("Please wait", "Prediction is running. Please stop it before closing.")  
        else:
            self.window.destroy()
    
    def _stop(self):
        self.stop_requested = True
        self.result_text.set("🛑 Stopping...")
        self.label_result.config(fg="red")
        self.stop_button.config(state="disabled")
        self.window.update()
    
    def check_stop(self):
        if self.stop_requested:
            self.logger.info("Prediction stopped by user.")
            self.success = False
            return True
        
    def _run(self):
        self.run_button.config(state="disabled")
        self.running = True
        mode = self.combo_modes.get()
        self.success = True
        self.stop_requested = False
        self.label_exec_mode.grid_remove()
        self.combo_modes.grid_remove()
        self.run_button.grid_remove()
        self.label_working_on.grid(row=6, column=0, pady=10)
        self.label_status.grid(row=7, column=0, pady=10)
        self.label_result.grid(row=8, column=0, columnspan=4, pady=10)
        self.stop_button.grid(row=5, column=2)
        if mode == "Prediction":
            self._run_prediction()
        else :
            self._run_bet()
    
    def _run_prediction(self):
        model_name = self.combo_models.get()
        if model_name != self.config.get("default","model"):
            self.config.set("default","model",model_name)
            self.config.save()
        if self.config.get("ModelChannels",model_name) == "2":
            self.option.set("flair",True)
        else : 
            self.option.set("flair",False)
        self.option.set("input_path",self.input_path.get())
        self.option.set('output_path',self.output_path.get())

        if self.open_viewer.get() : 
            viewer = self.combo_viewers.get()
            if viewer != self.config.get("default","viewer"):
                self.config.set("default","viewer",viewer)
                self.config.save()
        
        if self.save_bet.get():
            self.option.set("save_bet", True)
        else :
            self.option.set("save_bet", False)
        
        self._find_nii_files()
        if self.config.get("default","suffix")!=self.suffix.get():
            self.config.set("default","suffix", self.suffix.get())
            self.config.save()
        self.result_text.set("⌛ Prediction running...")
        self.label_result.config(fg="blue")
        t = threading.Thread(target=self._predict)
        t.start()



    def _predict(self):
        len_nii_paths= len(self.nii_paths)
        temp_dir = tempfile.mkdtemp(prefix="unet_preprocess")
        i=0
        self.logger.debug(f"paths dict : {self.nii_paths}")
        for t1, flair in self.nii_paths.items():
            try:
                if self.check_stop():
                    raise InterruptedError("Action was cancelled by the user.")
                s = f"Working on: {os.path.basename(t1)} ({i+1}/{len_nii_paths})"
                self.logger.info(f"Starting processing on: {t1}")
                self.window.after(0,self._update_stringvar,self.working_on_text,s)
                data, affine, bbox,original_shape, trsf_path, old_spacing, padding  = self.preprocessor.run(t1,flair,temp_dir)
                if self.check_stop():
                    raise InterruptedError("Action was cancelled by the user.")
                data = self.inference.run(data)
                if self.check_stop():
                    raise InterruptedError("Action was cancelled by the user.")
                if self.open_viewer.get() and i==0:
                    self.postprocessor.run(data,affine,t1,bbox,original_shape,temp_dir,trsf_path,old_spacing,padding,True)
                else : 
                    self.postprocessor.run(data,affine,t1,bbox,original_shape,temp_dir,trsf_path,old_spacing,padding)
                i+=1
            except Exception as e:
                self.logger.error(f"Erreur lors du traitement de {t1} : {e}")
                self.success = False
        self.window.after(0,self._update_result)
        # self.preprocessor.clean(temp_dir)

    def _run_bet(self):

        self.option.set("input_path",self.input_path.get())
        self.option.set('output_path',self.output_path.get())
        
        self._find_nii_files()

        self.result_text.set("⌛ Brain extraction running...")
        self.label_result.config(fg="blue")
        t = threading.Thread(target=self._bet)
        t.start()

    def _bet(self):
        len_nii_paths= len(self.nii_paths)
        temp_dir = tempfile.mkdtemp(prefix="unet_preprocess")
        i=0
        for img in self.nii_paths:
            try:
                if self.check_stop():
                    raise InterruptedError("Action was cancelled by the user.")
                s = f"Working on: {os.path.basename(img)} ({i+1}/{len_nii_paths})"
                self.window.after(0,self._update_stringvar,self.working_on_text,s)
                self.logger.info(f"Starting processing on: {img}")
                self.preprocessor.run(img,temp_dir,True)
                if self.check_stop():
                    raise InterruptedError("Action was cancelled by the user.")
                i+=1
            except Exception as e:
                self.logger.error(f"Erreur lors du traitement de {img} : {e}")
                self.success = False
        self.window.after(0,self._update_result)
        self.preprocessor.clean(temp_dir)

    def _update_result(self):
        self.label_working_on.grid_remove()
        self.label_status.grid_remove()
        self.stop_button.grid_remove()
        self.stop_button.config(state="normal")
        self.run_button.config(state="normal")
        self.label_exec_mode.grid(row=6, column=0)
        self.combo_modes.grid(row=6, column=1)
        self.run_button.grid(row=6,column=2)
        self.running = False
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

    def update_status(self,s):
        self.window.after(0,self._update_stringvar,self.status_text,s)

    def _update_stringvar(self,stringvar,s):
        stringvar.set(s)


    def _check_device(self):
        available_providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in available_providers:
            device = 'CUDAExecutionProvider'
        else : 
            device = 'CPUExecutionProvider'
        self.logger.info(f'using device : {device}')
        return device