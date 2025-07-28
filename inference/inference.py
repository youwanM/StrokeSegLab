import logging
import os
from gui.gui import GUIMain
import onnxruntime as ort
from utils.config_manager import Config
from utils.models_manager import update_models
from utils.option_manager import Option
import numpy as np
from scipy.ndimage import gaussian_filter
import time

from utils.path import MODEL_DIR

class Inference:
    """
    This class performs inference on 3D images using an ONNX model
    """
    def __init__(self,gui : GUIMain =None)->None:
        """
        Initialize the inference:
        - Setup de Config and Option class
        - Call for update_models to update the models list
        - Preload CUDA DLLs to avoid runtime errors when using the CUDAExecutionProvider
        Args:
            gui (GUIMain, optional): _description_. Defaults to None.
        """
        self.logger=logging.getLogger()
        option = Option()
        self.device = option.get("device")

        if self.device == "CUDAExecutionProvider":
            ort.preload_dlls(directory='')
        self.gui = gui

        self.config = Config()
        self.model_path = option.get("model_path")
        if self.model_path is None:
            update_models()
        
        self.patch_size = [128,128,128]

    
    def _compute_steps(self,image_size : tuple[int,int,int], patch_size : list[int], step_size : float =0.5)->list[list[int]]:
        """
        Compute the coordinates of starting positions for the inference
        Args:
            image_size (tuple[int,int,int]): Size of the image
            patch_size (list[int]): Size of the patch to extract
            step_size (float, optional): Controls the overlap between consecutive patches. Defaults to 0.5

        Returns:
            list[list[int]]: A list of three lists, each containing the start positions for the sliding window in the x, y, and z dimensions
        """
        assert [i >= j for i, j in zip(image_size, patch_size)], "image size must be as large or larger than patch_size"
        assert 0 < step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'

        # our step width is patch_size*step_size at most, but can be narrower. For example if we have image size of
        # 110, patch size of 64 and step_size of 0.5, then we want to make 3 steps starting at coordinate 0, 23, 46
        target_step_sizes_in_voxels = [i * step_size for i in patch_size]

        num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, patch_size)]

        steps = []
        for dim in range(len(patch_size)):
            # the highest step value for this dimension is
            max_step_value = image_size[dim] - patch_size[dim]
            if num_steps[dim] > 1:
                actual_step_size = max_step_value / (num_steps[dim] - 1)
            else:
                actual_step_size = 99999999999  # does not matter because there is only one step at 0

            steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]

            steps.append(steps_here)

        return steps
    
    def _compute_gaussian(self, patch_size, dtype=np.float32, sigma_scale=1./8, value_scaling_factor=10):
        tmp = np.zeros(patch_size)
        center_coords = [i // 2 for i in patch_size]
        sigmas = [i * sigma_scale for i in patch_size]
        tmp[tuple(center_coords)] = 1
        gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)

        gaussian_importance_map = gaussian_importance_map.astype(dtype)
        gaussian_importance_map /= (np.max(gaussian_importance_map) / value_scaling_factor)
        
        # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
        mask = gaussian_importance_map == 0
        gaussian_importance_map[mask] = np.min(gaussian_importance_map[~mask])
        return gaussian_importance_map

    def run(self,data):
        assert data.ndim == 4, f"Expected 4D input (c, x, y, z), got {data.shape}"
        _, x, y, z = data.shape
        if self.model_path is None : 
            model = self.config.get("default","model")
            self.model_path = os.path.join(MODEL_DIR,f"{model}.onnx")
        self.logger.debug(f'ONNX model path : {self.model_path}')
        
        # Set up ONNX Runtime providers based on device
        providers = [self.device]
        self.ort_session = ort.InferenceSession(self.model_path, providers=providers)
        self.logger.info("ONNX model loaded successfully.")
        steps = self._compute_steps((x, y, z),self.patch_size)
        gaussian = self._compute_gaussian(self.patch_size)
        total_patches = len(steps[0]) * len(steps[1]) * len(steps[2])
        count = 0
        output = np.zeros((1, 2, x, y, z))
        normalization_map = np.zeros((1, 1, x, y, z))
        start_time = time.time()
        last_time = start_time
        
        # Get ONNX model input name
        input_name = self.ort_session.get_inputs()[0].name
        
        for x_coord in steps[0]:
            for y_coord in steps[1]:
                for z_coord in steps[2]:
                    count+=1
                    if self.gui != None and self.gui.check_stop():
                        raise InterruptedError("Action was cancelled by the user.")
                    patch = data[:,x_coord:x_coord+self.patch_size[0],y_coord:y_coord+self.patch_size[1],z_coord:z_coord+self.patch_size[2]]
                    patch = np.expand_dims(patch, axis=0).astype(np.float32)
                    
                    # Run ONNX inference
                    pred = self.ort_session.run(None, {input_name: patch})[0]
                    pred = pred * gaussian
                    
                    output[:, :, x_coord:x_coord+self.patch_size[0], y_coord:y_coord+self.patch_size[1], z_coord:z_coord+self.patch_size[2]] += pred
                    normalization_map[:, :, x_coord:x_coord+self.patch_size[0], y_coord:y_coord+self.patch_size[1], z_coord:z_coord+self.patch_size[2]] += gaussian
                    
                    now = time.time()
                    total_elapsed = now - start_time
                    iter_time = now - last_time
                    remaining = (total_elapsed / count) * (total_patches - count)
                    self.logger.info(f"Patch {count}/{total_patches} done | " f"Iter time: {iter_time:.2f}s | " f"Total: {total_elapsed:.2f}s | " f"ETA: {remaining:.2f}s")
                    if self.gui !=None:
                        if self.gui.check_stop():
                            raise InterruptedError("Action was cancelled by the user.")
                        self.gui.update_status(f"Patch {count}/{total_patches} done | " f"Iter time: {iter_time:.2f}s | " f"Total: {total_elapsed:.2f}s | " f"ETA: {remaining:.2f}s")
                    last_time = now
        
        output /= normalization_map
        return output
