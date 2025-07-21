import logging
import shutil
import nibabel
import os
import numpy as np
import time

from manager.config_manager import Config
from manager.naming import DERIVATIVES
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
        self.config = Config()
        self.viewer = Viewer()
        self.gui = gui
    
    def _save_img(self,temp_dir,data,base_name,affine):
        start = time.time()
        out_img = nibabel.Nifti1Image(data,affine)
        suffix = self.config.get("default","suffix")
        output_file = os.path.join(temp_dir, base_name + f"_{suffix}.nii.gz")
        nibabel.save(out_img, output_file)
        return output_file, time.time()-start
    
    def _convert_to_segmentation(self, data):
        start = time.time()
        self.logger.debug(f"data shape : {data.shape}")
        data = data[0]
        self.logger.debug(f"data[0] shape : {data.shape}")
        for i in range(data.shape[0]):
            self.logger.debug(f"Stats for channel {i}: min={np.min(data[i])}, max={np.max(data[i])}, mean={np.mean(data[i]):.4f}, std={np.std(data[i]):.4f}, non-zero={np.count_nonzero(data[i])}")
        data = np.argmax(data, axis=0).astype(np.uint8)
        return data, time.time()-start
    
    def _register_to_reference(self,img_path,trsf_path,ref):
        start = time.time()
        xml_path = trsf_path.replace('.txt','.xml')
        command=["animaTransformSerieXmlGenerator","-i",trsf_path,"-o",xml_path]
        self.wrapper.run(command)

        command=["animaApplyTransformSerie","-i",img_path,"-t",xml_path,"-o",img_path,"-g",ref,"-I"]
        self.wrapper.run(command)
        return time.time()-start

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
    
    def _get_image_basename(self,img_path):
        filename = os.path.basename(img_path)
        if filename.endswith(".nii.gz"):
            return filename[:-7]
        elif filename.endswith(".nii"):
            return filename[:-4]
        else:
            return os.path.splitext(filename)[0]

    def move_to_output(self,img_path):
        subject_name = os.path.basename(img_path).split("_")[0]
        output_dir = os.path.join(self.option.get("input_path"),DERIVATIVES,subject_name,"anat")
        os.makedirs(output_dir,exist_ok=True)
        return shutil.copy(img_path,os.path.join(output_dir,os.path.basename(img_path)))


    def run(self,data,affine,input_path,bbox,original_shape,temp_dir,trsf_path,old_spacing,padding,bet,open_viewer=False):


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
        slicer = tuple(slice(start, end) for start, end in bbox)
        data = self._uncrop_from_bbox(data,slicer,original_shape)

        action_name="resampling"
        new_spacing = (1.0, 1.0, 1.0)
        self._print_action(action_name)
        data = np.expand_dims(data, axis=0)
        self.logger.debug(f'data dim : {data.shape}')
        data, time = self.resampler.run(data,new_spacing,old_spacing)
        data = data.squeeze(0)
        self._print_duration(action_name,time)

        basename = self._get_image_basename(input_path)
        if basename.endswith("_BET"):
            basename = basename[:-4]

        action_name="saving image to nii"
        self._print_action(action_name)
        nii_file, time = self._save_img(temp_dir,data,basename,affine)
        self._print_duration(action_name,time)

        action_name="register to reference"
        self._print_action(action_name)
        time = self._register_to_reference(nii_file,trsf_path,bet)
        self._print_duration(action_name,time)
        self.logger.debug(f"open viewer : {open_viewer}")

        output_path = self.move_to_output(nii_file)
        
        if open_viewer:
            action_name="open viewer"
            self._print_action(action_name)
            self.viewer.run(input_path,output_path)