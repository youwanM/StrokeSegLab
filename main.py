from inference.run_inference import Inference
from manager.option_manager import Option

inference = Inference("./model.pth","./sub-r001s001_0000.nii.gz",'./')
inference.run_inference()