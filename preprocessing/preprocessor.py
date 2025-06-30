import logging
from manager.option_manager import Option
from preprocessing.brain_extraction import BrainExtracter
from preprocessing.resampling import Resampler
from preprocessing.wrapper import AnimaWrapper
import time

import numpy as np
import nibabel
import tempfile
import os
import shutil
import nibabel as nib

class Preprocessor:
    def __init__(self,atlas_path="./anima_scripts/atlas.nrrd"):
        self.logger = logging.getLogger()
        self.option = Option()
        self.resampler = Resampler()
        self.wrapper = AnimaWrapper()
        self.temp_dir= tempfile.mkdtemp(prefix="unet_preprocess")
        self.atlas_path = atlas_path
        self.brain_extracter = BrainExtracter(self.wrapper,atlas_path)
    
    def _load_img(self,img_path):
        img = nibabel.load(img_path)
        affine = img.affine
        spacing = img.header.get_zooms()
        data = img.get_fdata().astype("float32")
        if data.ndim==3:
            data = np.expand_dims(data,axis=0)
        return data, spacing, affine
    
    def _z_score_norm(self,data, seg=None):
        start = time.time()
        if seg is not None :
            mask = seg >= 0
            mean = data[mask].mean()
            std = data[mask].std()
            data[mask] = (data[mask] - mean) / (max(std, 1e-8))
        else:
            mean = data.mean()
            std = data.std()
            data -= mean
            data /= (max(std, 1e-8))
        return data, start -time.time()

    def _bias_correct(self,img_path):
        start = time.time()
        output_path=img_path.replace('.nii.gz', '_N4.nii.gz')
        command=["animaN4BiasCorrection","-i",img_path,"-o",output_path]
        self.wrapper.run(command)
        return output_path, time.time()-start

    def _get_image_basename(self,img_path):
        filename = os.path.basename(img_path)
        if filename.endswith(".nii.gz"):
            return filename[:-7]
        elif filename.endswith(".nii"):
            return filename[:-4]
        else:
            return os.path.splitext(filename)[0]
        
    def _register_to_reference(self,img_path,ref):
        start = time.time()
        output_path= img_path.replace('.nii.gz', 'MNI.nii.gz')
        command=["animaPyramidalBMRegistration","-m",img_path,"-r",ref,"-o",output_path]
        self.wrapper.run(command)
        return output_path, time.time()-start

    def _print_duration(self,action_name,duration):
        self.logger.info(f"{action_name} took {duration:.2f} seconds.")

    def _print_action(self,action_name):
        self.logger.info(f"Starting {action_name}...")

    def _reorient_RAS(self,img_path):
        start = time.time()
        img = nib.load(img_path)
        reoriented_img = nib.as_closest_canonical(img)
        output_path=img_path.replace('.nii.gz','RAS.nii.gz')
        nib.save(reoriented_img, output_path)
        return output_path, time.time()-start

    
    def run(self,img_path):
        prefix = os.path.join(self.temp_dir,self._get_image_basename(img_path))

        action_name="brain extraction"
        self._print_action(action_name)
        masked_brain, time = self.brain_extracter.run(img_path,prefix)
        self._print_duration(action_name,time)

        action_name="bias correction"
        self._print_action(action_name)
        n4_output,time=self._bias_correct(masked_brain)
        self._print_duration(action_name,time)

        action_name="reorient to RAS"
        self._print_action(action_name)
        RAS_output,time=self._reorient_RAS(n4_output)
        self._print_duration(action_name,time)

        action_name="register to MNI"
        self._print_action(action_name)
        MNI_output,time=self._register_to_reference(RAS_output,self.atlas_path)
        self._print_duration(action_name,time)

        data, spacing, affine = self._load_img(MNI_output)
        new_spacing = (1.0, 1.0, 1.0)

        action_name="resampling"
        self._print_action(action_name)
        data, time = self.resampler.run(data,spacing,new_spacing)
        self._print_duration(action_name,time)

        action_name="Z score name"
        self._print_action(action_name)
        data, time = self._z_score_norm(data)
        self._print_duration(action_name,time)

        return data, affine


    def clean(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            self.logger.info(f"Temporary directory '{self.temp_dir}' has been removed.")