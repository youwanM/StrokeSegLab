import logging
import shutil
import nibabel
import os
import numpy as np
import time
from scipy.special import expit
from manager.config_manager import Config
from manager.naming import BET, DERIVATIVES, EXTENSIONS, MNI, RAWDATA, T1
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
    
    def _convert_to_segmentation(self, data,threshold):
        start = time.time()
        self.logger.debug(f"threshold : {threshold}")
        data = data[0]
        data = data[1]
        data = expit(data)
        # data = softmax(data,axis=0)
        # data = data[1]
        seg = (data >= threshold).astype(np.uint8)
        return seg, time.time()-start
    
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
        name = os.path.basename(img_path)
        for ext in EXTENSIONS:
            if name.endswith(ext):
                name = name[:-len(ext)]
        return name

    def move_to_output(self,img_path):
        subject_name = os.path.basename(img_path).split("_")[0]
        input_path = self.option.get("input_path")
        if self.option.get("is_file"):
            if RAWDATA in input_path :
                raw_dir = input_path.split(RAWDATA)[0]
                output_dir = os.path.join(raw_dir,DERIVATIVES,subject_name,"anat")
            else:
                output_dir = os.path.dirname(input_path)
        else :
            output_dir = os.path.join(input_path,DERIVATIVES,subject_name,"anat")
        os.makedirs(output_dir,exist_ok=True)
        return shutil.copy(img_path,os.path.join(output_dir,os.path.basename(img_path)))

    def _rm_entity(self,img_path,keyword):
        name = self._get_image_basename(img_path)
        i = name.find(keyword)
        if i ==-1:
            return name
        name = name[:i]
        if name.endswith("-"):
            name = name.rsplit("_",1)[0]
        name = name.rstrip("_")
        return name
    
    def run(self,data,affine,input_path,bbox,original_shape,temp_dir,trsf_path,old_spacing,padding,bet,MNI_base_image,threshold,open_viewer=False):


        action_name="convert to segmentation"
        self._print_action(action_name)
        data,time = self._convert_to_segmentation(data,threshold)
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

        basename = self._rm_entity(input_path,BET)
        if self.option.get("flair"):
            basename = self._rm_entity(basename,T1)

        action_name="saving image to nii"
        self._print_action(action_name)
        nii_file, time = self._save_img(temp_dir,data,basename,affine)
        self._print_duration(action_name,time)

        if trsf_path is None or self.option.get("keep_MNI"):
            new_output = self._get_image_basename(nii_file) + "_" + MNI + ".nii.gz"
            new_output = os.path.join(os.path.dirname(nii_file),new_output)
            nii_file = shutil.copy(nii_file,new_output)
            input_path = MNI_base_image
        else:
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