import logging
import sys
import tempfile
import tkinter as tk
from tkinter import Menu, filedialog, ttk, messagebox
import onnxruntime as ort
import os

from utils.string import APP_NAME, DEVELOPERS, HELP, LICENSE, PUBLICATIONS, VERSION
from inference.inference import Inference
from managers.config_manager import Config
from utils.models_manager import add_model, update_models
from managers.option_manager import Option
from utils.path import LOGO
from postprocessing.postprocessor import Postprocessor
from preprocessing.preprocessor import Preprocessor
import threading

class GUIMain:
    """
    Graphical interface of the brain segmentation application
    This class initialize the Tkinter window, manage user inputs, run preprocessing, inference, postprocessing and updates the GUI
    """
    def __init__(self, input_path : str,only_preprocessing : bool,save_preprocessing:bool, keep_MNI : bool ,save_pmap : bool ,threshold : float , model_name : str ,suffix : str ,viewer : str)->None:
        """
        Initialize the graphical interface with all the Tkinter widgets and run the main Tkinter loop (mainloop)

        Args:
            input_path (str): The input path
            only_preprocessing (bool): If True, the app will do the brain extraction only
            save_preprocessing (bool): If True, save all the preprocessing steps
            keep_MNI (bool): If True, the app will save the input image and the segmentation in the MNI space
            save_pmap (bool): If True, the app will save the probability map in addition to the binary mask
            threshold (float): Set the segmentation threshold to this value (0.5 if None)
            model_name (str): The model with this name located in the models folder will be used
            suffix (str): The suffix for the output segmentation (save it as default), use default one if None
            viewer (str): Name of the viewer used to open the first input image and its segmentation
            
        These parameters are used to pre-fill the corresponding fields in the graphical interface.
        """
        self.logger = logging.getLogger()
        self.logger.warning("="*60)
        self.logger.warning("This tool is for research purpose only ! ")
        self.logger.warning("="*60)
        self.option = Option() # A singleton used to store runtime option
        self.config = Config() # A singleton used to store configuration preferences and file paths using a .ini file
        self.option.set("device",self._check_device())
        self.preprocessor = Preprocessor(gui=self)
        self.postprocessor = Postprocessor(gui=self)
        self.inference = Inference(gui=self)
        self.running = False
        self.nii_paths = {}

        self.window = tk.Tk()
        self.window.protocol("WM_DELETE_WINDOW", self._on_close) # Call self._on_close when the user tries to close de window
        self.window.title(APP_NAME)
        if input_path is None : 
            self.input_path = tk.StringVar(value="")
        else :
            self.input_path = tk.StringVar(value=input_path)
        if suffix is None:
            self.suffix = tk.StringVar(value=self.config.get("default","suffix"))
        else :
            self.suffix = tk.StringVar(value=suffix)
        
        if viewer is None:
            self.open_viewer = tk.BooleanVar(value=False)
        else :
            self.open_viewer = tk.BooleanVar(value=True)
        
        self.keep_MNI = tk.BooleanVar(value=keep_MNI)

        self.save_pmap = tk.BooleanVar(value=save_pmap)
        self.save_preproc = tk.BooleanVar(value=save_preprocessing)

        self.channel_text = tk.StringVar()
        self.status_text = tk.StringVar()
        self.working_on_text = tk.StringVar()
        self.result_text = tk.StringVar()
        self.subject_number_text = tk.StringVar()
        if threshold is None:
            self.threshold_var = tk.DoubleVar(value=0.5)
        else : 
            self.threshold_var = tk.DoubleVar(value=threshold)

        menubar = Menu(self.window)
        
        self.option_menu = Menu(menubar,tearoff=0)
        self.option_menu.add_command(label='Threshold',command=self._show_threshold)
        self.option_menu.add_checkbutton(label="Save prerocessing", variable=self.save_preproc)
        self.option_menu.add_checkbutton(label="Save probability map", variable=self.save_pmap)
        self.option_menu.add_command(label='Import a model',command=self._show_import_model)
        self.option_menu.add_command(label="Restore warning window", command=self._restore_warning_window)
        menubar.add_cascade(label="Option",menu=self.option_menu)

        help_menu = Menu(menubar, tearoff=0)
        help_menu.add_command(label='Help', command=self._show_help)
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label='Help',menu=help_menu)
        self.window.config(menu=menubar)

        
        frame = tk.Frame(self.window)
        frame.grid()
        self.run_button = tk.Button(frame, text='run', command=self._run, state="disabled")
        self.run_button.grid(row=6,column=2)
        tk.Label(frame, text="Input path : ").grid(row=0,column=0,pady=10)
        self.entry_input_path = tk.Entry(frame, textvariable=self.input_path, state='readonly', width=50)
        self.entry_input_path.grid(row=0, column=1)
        tk.Button(frame, text='Select input folder',command=self._select_input_folder).grid(row=0,column=2)
        tk.Button(frame, text='Select input file',command=self._select_input_file).grid(row=0,column=3)

        self.label_suffix = tk.Label(frame, text="Suffix : ")
        self.entry_suffix = tk.Entry(frame, textvariable=self.suffix, width=20)

        self.label_model = tk.Label(frame, text="Model : ")
        self.models = self.config.get("default","models").split(',')
        self.models = [m for m in self.models if m !=""]

        self.label_channel = tk.Label(frame,textvariable=self.channel_text,fg="blue")
        self.combo_models = ttk.Combobox(frame,values=self.models,state="readonly")
        self.combo_models.bind("<<ComboboxSelected>>", self._on_model_change) # Call self._on_model_change when a model is selected
        default_model = self.config.get("default","model")
        if model_name is None:
            if default_model != "":
                self.combo_models.current(self.models.index(default_model))
        else :
            if model_name in self.models :
                self.combo_models.current(self.models.index(model_name))
            else : 
                self.logger.warning(f"Model '{model_name}' not in {self.models}")
                self.combo_models.current(self.models.index(default_model))
                
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
            if viewer is None or viewer == "default":
                self.combo_viewers.current(self.viewers.index(self.config.get("default","viewer")))
            else :
                try :
                    self.postprocessor.check_viewer(viewer)
                    self.combo_viewers.current(self.viewers.index(viewer))
                except Exception as e:
                    self.logger.warning(f"Viewer check failed: {e}")
                    self.combo_viewers.current(self.viewers.index(self.config.get("default","viewer")))

        self.label_keep_MNI = tk.Label(frame,text=" Output in MNI space ")
        self.keep_MNI_button = tk.Checkbutton(frame, text="YES/NO", variable=self.keep_MNI)

        execution_modes = ["Prediction", "Brain extraction only"]
        self.label_exec_mode = tk.Label(frame, text="Execution mode : ")
        self.label_exec_mode.grid(row=6, column=0)
        self.combo_modes = ttk.Combobox(frame,values=execution_modes,state="readonly")
        if only_preprocessing : 
            self.combo_modes.current(1)
        else:
            self.combo_modes.current(0)
        self.combo_modes.grid(row=6, column=1)
        self.combo_modes.bind("<<ComboboxSelected>>", self._on_mode_change) # Call self._on_mode_change when mode is selected
        self._on_mode_change()

        self.label_working_on = tk.Label(frame, textvariable=self.working_on_text, fg="blue")
        self.label_subject_number_text = tk.Label(frame, textvariable=self.subject_number_text, fg="blue")
        self.label_status = tk.Label(frame, textvariable=self.status_text)
        self.label_result = tk.Label(frame, textvariable=self.result_text, font=("Arial", 14, "bold"))
        self.stop_requested = False
        self.stop_button = tk.Button(frame, text="Stop", command=self._stop)

        if self.config.get("default","show_warning")=="1":
            self._show_warning()
        self.window.mainloop() # Starts the tkinter main loop in the main thread

    def _restore_warning_window(self):
        self.config.set("default","show_warning","1")
        self.option_menu.entryconfig("Restore warning window",state="disabled")
        self.config.save()

    def _show_warning(self):
        def on_ok():
            if var.get():
                self.config.set("default","show_warning","0")
                self.option_menu.entryconfig("Restore warning window",state="normal")
                self.config.save()
            else:
                self.option_menu.entryconfig("Restore warning window",state="disabled")
            waring_window.destroy()
        
        waring_window = tk.Toplevel(self.window)
        waring_window.title("⚠️ WARNING")
        waring_window.transient(self.window) # Make the warning window stay on top of the main window and minimize with it
        tk.Label(waring_window, text="This tool is for research purpose only!", wraplength=280, font=("Arial", 12)).pack(pady=10, padx=10)
        var = tk.BooleanVar()
        tk.Checkbutton(waring_window, text="Do not show again", variable=var).pack()
        tk.Button(waring_window, text="OK", command=on_ok).pack(pady=10)

    def _check_path_filled(self)->None:
        """
        Enable the run button only if the input path is set and there is model available
        """
        if self.input_path.get() and self.combo_models.get() != "":
            self.run_button.config(state='normal')
        else:
            self.run_button.config(state='disabled') 

    def _on_model_change(self,event : tk.Event =None)->None:
        """
        Update the app based on the selected model
        - Disbale the run button if there isn't a model available
        - Display an appropriate message based on the model
        Args:
            event (tk.Event, optional): The event triggered by the model selection. Defaults to None
        """
        model = self.combo_models.get()
        if model == "":
            self.channel_text.set("No model found")
            self.label_channel.config(fg="red")
            self.run_button.config(state='disabled') 
        else:
            self.label_channel.config(fg="blue")
            channels = self.config.get("ModelChannels",model)
            if channels == "2":
                self.channel_text.set("Using FLAIR and T1")
            else:
                self.channel_text.set("Using T1")
            if self.input_path.get():
                self.run_button.config(state='normal')

    def _on_mode_change(self,event: tk.Event=None)->None:
        """
        Update the app based on the selected mode:
        - Brain extraction only : all the fields are deleted execpt input path, some option are disable
        - Prediction : all the fields are displayed
        Args:
            event (tk.Event, optional): The event triggered by the mode selection. Defaults to None.
        """
        mode = self.combo_modes.get()
        if mode == "Prediction":
            self.label_suffix.grid(row=2, column=0, pady=10)
            self.entry_suffix.grid(row=2, column=1)
            self.label_model.grid(row=3, column=0, pady=10)
            self.combo_models.grid(row =3, column =1)
            self.label_channel.grid(row=3,column=2)
            self.label_open_viewer.grid(row=4, column=0, pady=10)
            self.viewer_button.grid(row=4,column=1)
            self.label_keep_MNI.grid(row=5,column=0)
            self.keep_MNI_button.grid(row=5,column=1)
            self.option_menu.entryconfig("Save probability map",state="normal")
            self.option_menu.entryconfig("Save prerocessing",state="normal")
            self.option_menu.entryconfig("Threshold",state="normal")
            if len(self.viewers)==0:
                self.label_viewer_not_found.grid(row=4, column=2)
            else : 
                self.combo_viewers.grid(row=4, column=2)
        else : 
            self.label_suffix.grid_remove()
            self.entry_suffix.grid_remove()
            self.label_model.grid_remove()
            self.label_open_viewer.grid_remove()
            self.viewer_button.grid_remove()
            self.label_channel.grid_remove()
            self.combo_models.grid_remove()
            self.label_keep_MNI.grid_remove()
            self.keep_MNI_button.grid_remove()
            self.save_pmap.set(False)
            self.save_preproc.set(False)
            self.option_menu.entryconfig("Save probability map",state="disabled")
            self.option_menu.entryconfig("Save prerocessing",state="disabled")
            self.option_menu.entryconfig("Threshold",state="disabled")
            if len(self.viewers)==0:
                self.label_viewer_not_found.grid_remove()
            else : 
                self.combo_viewers.grid_remove()
            
    def _show_threshold(self)->None:
        """
        Display the threshold window with a slider and an entry field linked to the threshold of the segmentation
        """
        threshold_window= tk.Toplevel(self.window)
        scale = tk.Scale(threshold_window,orient=tk.HORIZONTAL,from_=0, to=1, resolution=0.01, label="Threshold", variable=self.threshold_var, length=200)
        scale.pack()

        entry_label = tk.Label(threshold_window, text="Enter value (0.0 - 1.0):")
        entry_label.pack()
        entry = tk.Entry(threshold_window, width=6, justify='center')
        entry.insert(0, str(self.threshold_var.get()))
        entry.pack()
        
        # Synchronize Entry with Scale
        def on_entry_change(*args):
            try:
                value = float(entry.get())
                if 0.0 <= value <= 1.0:
                    self.threshold_var.set(round(value, 2))
            except ValueError:
                pass
        
        # Synchronize Scale with Entry
        def on_scale_change(value):
            entry.delete(0, tk.END)
            entry.insert(0, str(round(float(value), 2)))

        entry.bind("<KeyRelease>", lambda event: on_entry_change())
        scale.config(command=on_scale_change)

        ok_button = tk.Button(threshold_window, text="OK", command=threshold_window.destroy)
        ok_button.pack(pady=10)

    def _show_about(self)->None:
        """
        Display the about window
        A notebook is used to display 3 frame in different tabs : developpers, license and publications
        """
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

    def _show_help(self)->None:
        """
        Display the help window
        """
        size =500
        help_window= tk.Toplevel(self.window)
        tk.Label(help_window, text=APP_NAME, font=("Arial", 18, "bold")).pack(pady=10)
        tk.Label(help_window,text=VERSION).pack(pady=10)
        help_window.logo = tk.PhotoImage(file=LOGO)
        tk.Label(help_window, text = HELP, justify='left',wraplength=size ).pack(anchor='w')
        tk.Button(help_window, text="Close", command=help_window.destroy).pack(anchor='e',padx=10,pady=[0,10])

    def _show_import_model(self)->None:
        """
        Display the import model window with 3 button to select and import the model and close the window
        """
        import_model_window= tk.Toplevel(self.window)
        import_model_window.transient(self.window) # Make the import window stay on top of the main window and minimize with it
        import_model_window.title("Import Model")
        self.model_to_import = tk.StringVar(value="")
        self.status_import_text = tk.StringVar(value="")
        tk.Entry(import_model_window, textvariable=self.model_to_import, state='readonly', width=50).pack(padx=10, pady=5)
        self.label_import_model = tk.Label(import_model_window, textvariable=self.status_import_text)
        self.label_import_model.pack(padx=10, pady=5)
        button_frame = tk.Frame(import_model_window)
        button_frame.pack(padx=10, pady=10)

        tk.Button(button_frame, text='Select Model', command=self._select_model).pack(side='left', padx=5)
        tk.Button(button_frame, text='Import', command=self._import_model).pack(side='left', padx=5)
        tk.Button(button_frame, text='Close', command=import_model_window.destroy).pack(side='left', padx=5)

    def _select_model(self)->None:
        """
        Open a filedialog to select a model file.
        If a file is selected, store the path and clear the import status message
        """
        filename = filedialog.askopenfilename(title='Select model file')
        if filename:
            self.model_to_import.set(filename)
            self.status_import_text.set("")
    
    def _import_model(self)->None:
        """
        Try to import a model, update the model list and the status message : 
        - Call update_model to refresh the list of model available
        - Call add_model to add the model to the models directory
        - Update the list used by the combobox
        - Handle the potential error and display a status message
        """
        try:
            update_models()
            model_name = add_model(self.model_to_import.get())
            s = f"The model {model_name} was successfully imported"
            self.logger.info(s)
            self.status_import_text.set(s)
            self.label_import_model.config(fg="green")
            update_models()
            self.models = [m.strip() for m in self.config.get("default", "models").split(",")]
            self.combo_models['values'] = self.models
        except ValueError as e:
            s=f"Import failed: {e}"
            self.logger.error(s)
            self.status_import_text.set(s)
            self.label_import_model.config(fg="red")
        except Exception as e:
            s = f"Unexpected error during model import: {e}"
            self.logger.error(s)
            self.status_import_text.set(s)
            self.label_import_model.config(fg="red")

    def _select_input_folder(self)->None:
        """
        Handle the input folder selection
        Open a filedialog, set the input path and call for _check_path_filled
        """
        self.input_path.set(filedialog.askdirectory(title='input path'))
        self._check_path_filled()

    def _select_input_file(self)->None:
        """
        Handle the input file selection
        Open a filedialog, set the input path and call for _check_path_filled
        """
        self.input_path.set(filedialog.askopenfilename(title='input path'))
        self._check_path_filled()

    def _on_close(self)->None:
        """
        Check if a prediction is running, and destroy the window if not
        """
        if self.running:
            messagebox.showwarning("Please wait", "Prediction is running. Please stop it before closing.")  
        else:
            self.window.destroy()
    
    def _stop(self)->None:
        """
        Method used to stop a prediction (or BET)
        - Since the prediction (or BET) runs in a separate thread and can take a long time, this method does not stop the process immediately. Instead, it sets a variable (self.stop_requested) that is checked periodically during the processing loop. When detected, the process will terminate.
        - Update the UI to inform the user
        """
        self.stop_requested = True
        self.result_text.set("🛑 Stopping...")
        self.label_result.config(fg="red")
        self.stop_button.config(state="disabled")
        self.window.update()
    
    def check_stop(self) -> bool:
        """
        The method used during the process to check if a stop is requested by the user

        Returns:
            bool: True if a stop was requested, otherwise False
        """
        if self.stop_requested:
            self.logger.info("Prediction stopped by user.")
            self.success = False
            return True
        
    def _run(self)->None:
        """
        The entry point of the processing
        Update the UI, set various variables used during the processing task, and call either _run_prediction or _run_bet depending on the selected mode.
        """
        self.run_button.config(state="disabled")
        self.running = True
        mode = self.combo_modes.get()
        self.success = True
        self.stop_requested = False
        self.label_exec_mode.grid_remove()
        self.combo_modes.grid_remove()
        self.run_button.grid_remove()
        self.label_subject_number_text.grid(row=6, column=0, padx=10)
        self.label_working_on.grid(row=6, column=1, pady=10)
        self.label_status.grid(row=7, column=0, pady=10)
        self.label_result.grid(row=8, column=0, columnspan=4, pady=10)
        self.stop_button.grid(row=5, column=2)
        if mode == "Prediction":
            self._run_prediction()
        else :
            self._run_bet()
    
    def _run_prediction(self)->None:
        """
        Configure the prediction settings and launch it in a separate thread in order to not freeze the UI :
        - Retrieves and configures runtime option
        - Call find_nii_files to construct a dictionary of NIFTI path and check T1/FLAIR consistency
        - Display warning message if needed and other informations to the users
        - Start the prediction in a thread
        """
        model_name = self.combo_models.get()
        if model_name != self.config.get("default","model"):
            self.config.set("default","model",model_name)
            self.config.save()
        if self.config.get("ModelChannels",model_name) == "2":
            self.option.set("flair",True)
        else : 
            self.option.set("flair",False)
        self.option.set("input_path",self.input_path.get())

        if self.open_viewer.get() : 
            viewer = self.combo_viewers.get()
            if viewer != self.config.get("default","viewer"):
                self.config.set("default","viewer",viewer)
                self.config.save()
        
        self.option.set("save_pmap", self.save_pmap.get())
        self.option.set("save_preproc", self.save_preproc.get())
        self.option.set("keep_MNI", self.keep_MNI.get())
        
        self.nii_paths,subject,flair,none_list = self.preprocessor.find_nii_files()
        if self.option.get("flair") and subject != flair:
            if none_list:
                none_string = ", ".join(none_list)
                m = f"These subjects : \"{none_string}\" are missing either a T1 or a FLAIR image."
                self.window.after(0, lambda: messagebox.showwarning("⚠️ Warning", m))
            s = f"{len(self.nii_paths)} subject(s) found, \n ⚠️ T1 ({subject}) and FLAIR ({flair}) count mismatch"
            self._update_stringvar(self.subject_number_text,s)
        else:
            s = f"{subject} subject(s) found"
            self._update_stringvar(self.subject_number_text,s)
                

        if self.config.get("default","suffix")!=self.suffix.get():
            self.config.set("default","suffix", self.suffix.get())
            self.config.save()
        self.result_text.set("⌛ Prediction running...")
        self.label_result.config(fg="blue")
        t = threading.Thread(target=self._predict) 
        t.start() # Start a thread that calls for self._predict



    def _predict(self)->None:
        """
        Run the prediction pipeline for all subjects.

        Depending on the model:
        - 2 channels (FLAIR and T1), 't1' is the T1 image and 'flair' is the FLAIR image.
        - 1 channel, 't1' contains either the T1 or the FLAIR image, and 'flair' is None.

        It creates a temporary directory and then, for each subject : 
        - Run preprocessing, inference and postprocessing
        - Checks periodically if a stop is requested
        - Update the UI during the process using self.window.after because Tkinter isn't thread safe, we can't modify the UI from a secondary thread. So 'after' is scheduling an update in the main thread
        """
        threshold = self.threshold_var.get()
        len_nii_paths= len(self.nii_paths)
        temp_dir = tempfile.mkdtemp(prefix="unet_preprocess")
        i=0
        for t1, flair in self.nii_paths.items():
            try:
                if self.check_stop():
                    raise InterruptedError("Action was cancelled by the user.")
                if flair is None:
                    s = f"Working on: {os.path.basename(t1)} ({i+1}/{len_nii_paths})"
                    self.logger.info(f"Starting processing on: {os.path.basename(t1)}")
                else :
                    s=f"Starting processing on: ({os.path.basename(t1)},{os.path.basename(flair)})"
                    self.logger.info(f"Starting processing on: ({os.path.basename(t1)},{os.path.basename(flair)})")
                self.window.after(0,self._update_stringvar,self.working_on_text,s) # We schedule a method that will be executed in the main thread
                data, affine, bbox,original_shape, trsf_path, old_spacing, padding, bet, MNI_base_image  = self.preprocessor.run(t1,flair,temp_dir)
                if self.check_stop():
                    raise InterruptedError("Action was cancelled by the user.")
                data = self.inference.run(data)
                if self.check_stop():
                    raise InterruptedError("Action was cancelled by the user.")
                if self.open_viewer.get() and i==0: # Only open the viewer for the first image
                    self.postprocessor.run(data,affine,t1,bbox,original_shape,temp_dir,trsf_path,old_spacing,padding,bet,MNI_base_image,threshold,True)
                else : 
                    self.postprocessor.run(data,affine,t1,bbox,original_shape,temp_dir,trsf_path,old_spacing,padding,bet,MNI_base_image,threshold)
                i+=1
            except Exception as e: # If an exception occurs, processing of the current subject is stopped, a message is logged to inform the user, and the process continues with the next subject
                self.logger.error(f"Erreur lors du traitement de {t1} : {e}")
                self.success = False
        self.window.after(0,self._update_result) # Call for update_result in the main thread
        self.preprocessor.clean(temp_dir)

    def _run_bet(self)->None:
        """
        Configure the BET settings and launch it in a separate thread in order to not freeze the UI
        """
        self.option.set("input_path",self.input_path.get())
        
        self.nii_paths,*_ = self.preprocessor.find_nii_files()

        self.result_text.set("⌛ Brain extraction running...")
        self.label_result.config(fg="blue")
        t = threading.Thread(target=self._bet) 
        t.start() # Start a thread that calls for self._bet

    def _bet(self)->None:
        """
        Run the BET pipeline for all subjects.
        
        It creates a temporary directory and then, for each subject : 
        - Run preprocessing
        - Checks periodically if a stop is requested
        - Update the UI during the process using self.window.after because Tkinter isn't thread safe, we can't modify the UI from a secondary thread. So 'after' is scheduling an update in the main thread
        """
        len_nii_paths= len(self.nii_paths)
        temp_dir = tempfile.mkdtemp(prefix="unet_preprocess")
        i=0
        for t1, flair in self.nii_paths.items():
            try:
                if self.check_stop():
                    raise InterruptedError("Action was cancelled by the user.")
                s = f"Working on: {os.path.basename(t1)} ({i+1}/{len_nii_paths})"
                self.logger.info(f"Starting processing on: {t1}")
                self.window.after(0,self._update_stringvar,self.working_on_text,s) # Call for _update_stringvar in the main thread
                self.preprocessor.run(t1,flair,temp_dir,True)
                if self.check_stop():
                    raise InterruptedError("Action was cancelled by the user.")
                i+=1
            except Exception as e: # If an exception occurs, processing of the current subject is stopped, a message is logged to inform the user, and the process continues with the next subject
                self.logger.error(f"Erreur lors du traitement de {t1} : {e}")
                self.success = False
        self.window.after(0,self._update_result) # Call for update_result in the main thread
        self.preprocessor.clean(temp_dir)

    def _update_result(self)->None:
        """
        Update the UI based on the result of the processing
        """
        self.label_working_on.grid_remove()
        self.label_status.grid_remove()
        self.stop_button.grid_remove()
        self.stop_button.config(state="normal")
        self.run_button.config(state="normal")
        self.label_exec_mode.grid(row=6, column=0)
        self.combo_modes.grid(row=6, column=1)
        self.run_button.grid(row=6,column=2)
        self.label_subject_number_text.grid_remove()
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

    def _update_stringvar(self,stringvar : tk.StringVar,s: str)->None:
        """
        Update the value of a StringVar with the given string
        Args:
            stringvar (tk.StringVar): The StringVar to update
            s (str): The new string value to set
        """
        stringvar.set(s)


    def _check_device(self)->str:
        """
        Check Cuda if available

        Returns:
            str: The selected execution provider ('CUDAExecutionProvider' or 'CPUExecutionProvider')
        """
        available_providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in available_providers:
            device = 'CUDAExecutionProvider'
        else : 
            device = 'CPUExecutionProvider'
        self.logger.info(f'using device : {device}')
        return device