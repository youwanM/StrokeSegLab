import tkinter as tk
import sys
import threading
from entrypoints.cli import CLIMain # Import your CLI class

class GUIMain:
    def __init__(self, args):
        self._window = tk.Tk() 
        self.args = args
        self.threshold_var = tk.DoubleVar(value=args.threshold)

    def on_closing(self):
        """The OpenApps Handshake: Triggered by syngo.via when the user closes the study"""
        print("Received close signal from syngo.via. Shutting down gracefully.")
        self._window.destroy()
        sys.exit(0) # Pass '0' (Success) back to the C++ bootstrapper

    def _run_segmentation_thread(self):
        """Runs the actual segmentation in a background thread"""
        try:
            # 1. Update the arguments object with the slider's current value
            self.args.threshold = self.threshold_var.get()
            
            # 2. Instantiate and run the CLI logic directly!
            print(f"Starting segmentation with threshold: {self.args.threshold}")
            cli_app = CLIMain(self.args)
            cli_app.run()
            
            # 3. Safely update the GUI from the background thread
            self._window.after(0, self._processing_finished)
        except Exception as e:
            print(f"Segmentation failed: {e}")
            self._window.after(0, self._processing_failed, str(e))

    def _processing_finished(self):
        """Updates UI when PyTorch is done"""
        self.apply_btn.config(text="✅ Processing Complete! (Close study in syngo.via)", bg="darkgreen")

    def _processing_failed(self, error_msg):
        """Updates UI if an error occurs"""
        self.apply_btn.config(text=f"❌ Error: {error_msg[:20]}...", bg="red")

    def run_frontier_config(self):
        """The minimal Configuration UI for syngo.via Frontier"""
        # 1. WINDOW TITLE: Must match the static title in the Container Wizard
        self._window.title("StrokeSeg")
        self._window.geometry("400x200")

        # 2. TITLE BAR SUPPRESSION: Hide borders if syngo.via is embedding us
        if "-disable-title-bar" in sys.argv:
            self._window.overrideredirect(True)

        # 3. WM_CLOSE HANDLER: Bind the syngo.via close event
        self._window.protocol("WM_DELETE_WINDOW", self.on_closing)

        tk.Label(self._window, text="Adjust Segmentation Threshold", font=("Arial", 12, "bold")).pack(pady=10)
        
        # Threshold Slider
        scale = tk.Scale(self._window, orient=tk.HORIZONTAL, from_=0, to=1, 
                         resolution=0.01, variable=self.threshold_var, length=300)
        scale.pack(pady=10)

        def apply_and_start():
            """Disables the button and starts the background thread"""
            # Update UI immediately
            self.apply_btn.config(text="Processing in background... Please wait.", state=tk.DISABLED, bg="grey")
            
            # Start the heavy lifting in a thread so the GUI doesn't freeze
            threading.Thread(target=self._run_segmentation_thread, daemon=True).start()

        self.apply_btn = tk.Button(self._window, text="Apply & Start Segmentation", 
                                   command=apply_and_start, bg="green", fg="white")
        self.apply_btn.pack(pady=20)
        
        self._window.mainloop()