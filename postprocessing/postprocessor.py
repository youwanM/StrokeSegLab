import logging
import nibabel
import os
import numpy as np
import time

from manager.option_manager import Option
from postprocessing.viewer import Viewer
from preprocessing.resampling import Resampler
from preprocessing.wrapper import AnimaWrapper

class Postprocessor:
    def __init__(self,gui=None):
        self.option = Option()
        self.logger =logging.getLogger()
        self.wrapper = AnimaWrapper()
        self.resampler = Resampler()
        if self.option.get("viewer"):
            self.viewer = Viewer()
        self.gui = gui
    
    def _save_img(self,temp_dir,data,input_path,affine):
        start = time.time()
        out_img = nibabel.Nifti1Image(data,affine)
        base_name = os.path.basename(input_path)
        base_name = base_name.split(".nii")[0]
        suffix = self.option.get("suffix")
        output_file = os.path.join(temp_dir, base_name + f"_{suffix}.nii.gz")
        nibabel.save(out_img, output_file)
        return output_file, time.time()-start
    
    def _convert_to_segmentation(self, data):
        start = time.time()
        data = data[0]
        for i in range(data.shape[0]):
            self.logger.debug(f"Stats for channel {i}: min={np.min(data[i])}, max={np.max(data[i])}, mean={np.mean(data[i]):.4f}, std={np.std(data[i]):.4f}, non-zero={np.count_nonzero(data[i])}")
        data = np.argmax(data, axis=0).astype(np.uint8)
        return data, time.time()-start
    
    def _register_to_reference(self,img_path,trsf_path,ref):
        start = time.time()
        xml_path = trsf_path.replace('.txt','.xml')
        command=["animaTransformSerieXmlGenerator","-i",trsf_path,"-o",xml_path]
        self.wrapper.run(command)

        output_path= os.path.join(self.option.get("output_path"),os.path.basename(img_path))
        command=["animaApplyTransformSerie","-i",img_path,"-t",xml_path,"-o",output_path,"-g",ref,"-I"]
        self.wrapper.run(command)
        return output_path, time.time()-start

    def _print_duration(self,action_name,duration):
        self.logger.info(f"{action_name} took {duration:.2f} seconds.")

    def _print_action(self,action_name):
        self.logger.info(f"Starting {action_name}...")
        if(self.gui !=None):
            self.gui.update_status(f"Postprocessing : Starting {action_name}...")
    
    def _remove_padding(self,data, padding):
        start = time.time()
        slices = []
        for dim_pad in padding:
            start = dim_pad[0]
            end = -dim_pad[1] if dim_pad[1] > 0 else None
            slices.append(slice(start, end))
        return data[tuple(slices)], time.time()-start

    def _uncrop_from_bbox(self,data,bbox,original_shape):
        full_volume = np.zeros(original_shape, dtype=data.dtype)
        full_volume[bbox]=data
        full_volume = np.transpose(full_volume, (2, 1, 0))
        return full_volume
    
    def check_viewer(self, viewer):
        self.viewer.check_viewer(viewer)


    def run(self,data,affine,input_path,bbox,original_shape,temp_dir,trsf_path,old_spacing,padding,open_viewer=False):


        action_name="convert to segmentation"
        self._print_action(action_name)
        data,time = self._convert_to_segmentation(data)
        self._print_duration(action_name,time)

        action_name="remove padding"
        self._print_action(action_name)
        data,time = self._remove_padding(data,padding)
        self._print_duration(action_name,time)

        action_name="uncrop"
        self._print_action(action_name)
        data = self._uncrop_from_bbox(data,bbox,original_shape)

        action_name="resampling"
        new_spacing = (1.0, 1.0, 1.0)
        self._print_action(action_name)
        data = np.expand_dims(data, axis=0)
        self.logger.debug(f'data dim : {data.shape}')
        data, time = self.resampler.run(data,new_spacing,old_spacing)
        data = data.squeeze(0)
        self._print_duration(action_name,time)

        action_name="saving image to nii"
        self._print_action(action_name)
        nii_file, time = self._save_img(temp_dir,data,input_path,affine)
        self._print_duration(action_name,time)

        action_name="register to reference"
        self._print_action(action_name)
        output_path,time = self._register_to_reference(nii_file,trsf_path,input_path)
        self._print_duration(action_name,time)

        if open_viewer:
            action_name="open viewer"
            self._print_action(action_name)
            self.viewer.run(input_path,output_path)