import tkinter as tk
import os

class GUIMain:
    def __init__(self, args):
        # Initialize the window first for Tkinter variables
        self._window = tk.Tk() 
        self.args = args
        self.threshold_var = tk.DoubleVar(value=args.threshold)
        
        # You can still access other args like this:
        # self.skip_bet = args.skip_bet

    def run_frontier_config(self):
        """The minimal Configuration UI for syngo.via Frontier"""
        self._window.title("Segmentation Settings")
        self._window.geometry("400x200")

        tk.Label(self._window, text="Adjust Segmentation Threshold", font=("Arial", 12, "bold")).pack(pady=10)
        
        # Threshold Slider
        scale = tk.Scale(self._window, orient=tk.HORIZONTAL, from_=0, to=1, 
                         resolution=0.01, variable=self.threshold_var, length=300)
        scale.pack(pady=10)

        def save_and_close():
            # Save value to {SuspendDirectory} for the console process
            if hasattr(self.args, 'session_dir') and self.args.session_dir:
                os.makedirs(self.args.session_dir, exist_ok=True)
                path = os.path.join(self.args.session_dir, "threshold.txt")
                with open(path, "w") as f:
                    f.write(str(self.threshold_var.get()))
                print(f"Threshold saved to: {path}") # Debug log
            self._window.destroy()

        tk.Button(self._window, text="Apply & Start Segmentation", 
                  command=save_and_close, bg="green", fg="white").pack(pady=20)
        
        self._window.mainloop()