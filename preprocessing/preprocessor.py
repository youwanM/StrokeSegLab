import logging
from manager.config_manager import Config
from manager.naming import BET, DERIVATIVES, EXTENSIONS, FLAIR, RAWDATA, T1
from manager.option_manager import Option
from preprocessing.brain_extraction import BrainExtracter
from preprocessing.resampling import Resampler
from preprocessing.wrapper import AnimaWrapper
from scipy.ndimage import binary_fill_holes
from preprocessing.utils import get_bbox_from_mask, bounding_box_to_slice
import time
from manager.path import ATLAS_DIR

import numpy as np
import nibabel
import os
import shutil
import nibabel as nib

class Preprocessor:
    def __init__(self,postprocessor,gui=None):
        self.postprocessor = postprocessor
        self.logger = logging.getLogger()
        self.option = Option()
        self.config = Config()
        self.resampler = Resampler()
        self.wrapper = AnimaWrapper()
        self.atlas_path = os.path.join(ATLAS_DIR,"atlas.nrrd")
        self.gui = gui
        self.brain_extracter = BrainExtracter(self.wrapper,gui)
    
    def _load_img(self,img_path):
        img = nibabel.load(img_path)
        affine = img.affine
        spacing = img.header.get_zooms()
        data = img.get_fdata().astype("float32")
        data = np.transpose(data, (2, 1, 0))
        original_shape=data.shape
        if data.ndim==3:
            data = np.expand_dims(data,axis=0)
        return data, spacing, affine, original_shape
    
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
        name = os.path.basename(img_path)
        for ext in EXTENSIONS:
            if name.endswith(ext):
                name = name[:-len(ext)]
        return name
        
    def _register_to_reference(self,img_path,ref,suffix,prefix=None):
        start = time.time()
        if prefix is None:
            if img_path.endswith(".nii.gz"):
                output_path= img_path.replace('.nii.gz', "_"+suffix+'.nii.gz')
            elif img_path.endswith(".nii"):
                output_path= img_path.replace('.nii', "_"+suffix+'.nii.gz')
        else : 
            output_path = prefix + "_" + suffix + ".nii.gz"
        trsf_path = output_path.replace('.nii.gz', '.txt')
        command=["animaPyramidalBMRegistration","-m",img_path,"-r",ref,"-o",output_path,"-O",trsf_path]
        self.wrapper.run(command)
        return output_path,trsf_path, time.time()-start

    def _print_duration(self,action_name,duration):
        self.logger.info(f"{action_name} took {duration:.2f} seconds.")

    def _print_action(self,action_name):
        self.logger.info(f"Starting {action_name}...")
        if(self.gui !=None):
            self.gui.update_status(f"Preprocessing : Starting {action_name}...")

    def _reorient_RAS(self,img_path):
        start = time.time()
        img = nib.load(img_path)
        reoriented_img = nib.as_closest_canonical(img)
        output_path=img_path.replace('.nii.gz','RAS.nii.gz')
        nib.save(reoriented_img, output_path)
        return output_path, time.time()-start


    def _create_nonzero_mask(self,data):

        assert data.ndim in (3, 4), "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
        nonzero_mask = data[0] != 0
        for c in range(1, data.shape[0]):
            nonzero_mask |= data[c] != 0
        return binary_fill_holes(nonzero_mask)


    def _crop_to_nonzero(self,data, seg=None, nonzero_label=-1,bbox=None):
        nonzero_mask = self._create_nonzero_mask(data)
        if bbox is None:
            bbox = get_bbox_from_mask(nonzero_mask)
        slicer = bounding_box_to_slice(bbox)
        nonzero_mask = nonzero_mask[slicer][None]
        slicer = (slice(None), ) + slicer
        data = data[slicer]
        if seg is not None:
            seg = seg[slicer]
            seg[(seg == 0) & (~nonzero_mask)] = nonzero_label
        else:
            seg = np.where(nonzero_mask, np.int8(0), np.int8(nonzero_label))
        return data, seg, bbox

    def _padding(self,data, min_size=128):
        start = time.time()
        padding = []
        if data.ndim == 4 and data.shape[0] == 1:
            data = np.squeeze(data, axis=0)
        for dim in data.shape:
            total_pad = max(0, min_size - dim)
            pad_before = total_pad // 2
            pad_after = total_pad - pad_before
            padding.append((pad_before, pad_after))
        
        padded_data = np.pad(data, padding, mode='constant', constant_values=0)
        padded_data = np.expand_dims(padded_data, axis=0)
        return padded_data, padding, time.time()-start

    def run(self,t1,flair,temp_dir,bet_only=False):
        
        prefix = os.path.join(temp_dir,self._get_image_basename(t1))
        if not prefix.endswith("_BET"):
            if self.gui != None and self.gui.check_stop():
                raise InterruptedError("Action was cancelled by the user.")
            action_name="brain extraction"
            self._print_action(action_name)
            bet_t1, time = self.brain_extracter.run(t1,prefix)
            self._print_duration(action_name,time)
            if self.gui != None and self.gui.check_stop():
                raise InterruptedError("Action was cancelled by the user.")
        else :
            bet_t1 = shutil.copy(t1,temp_dir)
        if not bet_only:
            data_t1, affine, bbox, original_shape, trsf_path, spacing, padding = self._preprocess_modality(bet_t1)
            if self.option.get("save_bet"):
                self.postprocessor.move_to_output(bet_t1)
        else : 
            self.postprocessor.move_to_output(bet_t1)

        if self.option.get("flair"):
            if flair is not None:
                prefix = os.path.join(temp_dir,self._get_image_basename(flair))
                if not prefix.endswith("_BET"):
                    if self.gui != None and self.gui.check_stop():
                        raise InterruptedError("Action was cancelled by the user.")
                    action_name = "register FLAIR to T1"
                    self._print_action(action_name)
                    flair_t1,_, time = self._register_to_reference(flair, t1,"T1",prefix=prefix)
                    self._print_duration(action_name, time)
                    if self.gui != None and self.gui.check_stop():
                        raise InterruptedError("Action was cancelled by the user.")
                    action_name="brain extraction"
                    self._print_action(action_name)
                    bet_flair, time = self.brain_extracter.run(flair_t1,prefix)
                    self._print_duration(action_name,time)
                    if self.gui != None and self.gui.check_stop():
                        raise InterruptedError("Action was cancelled by the user.")
                else :
                    bet_flair = shutil.copy(flair,temp_dir)
                if not bet_only:
                    data_flair, *_ = self._preprocess_modality(bet_flair,bbox=bbox)
                    data = np.concatenate([data_t1, data_flair], axis=0)
                    if self.option.get("save_bet"):
                        self.postprocessor.move_to_output(bet_flair)
                    return data, affine, bbox, original_shape, trsf_path, spacing, padding, bet_t1
                else : 
                    self.postprocessor.move_to_output(bet_flair)
            else:
                raise ValueError("FLAIR image is required but was not provided")
        elif not bet_only:
            return data_t1, affine, bbox, original_shape, trsf_path, spacing, padding, bet_t1

    def _print_shape(self,data):
        self.logger.debug(f'data shape : {data.shape}')
    
    def _preprocess_modality(self,modality,bbox = None):
        action_name="bias correction"
        self._print_action(action_name)
        n4_output,time=self._bias_correct(modality)
        self._print_duration(action_name,time)
        if self.gui != None and self.gui.check_stop():
            raise InterruptedError("Action was cancelled by the user.")

        action_name="reorient to RAS"
        self._print_action(action_name)
        RAS_output,time=self._reorient_RAS(n4_output)
        self._print_duration(action_name,time)
        if self.gui != None and self.gui.check_stop():
            raise InterruptedError("Action was cancelled by the user.")
        
        action_name="register to MNI"
        self._print_action(action_name)
        MNI_output,trsf_path,time=self._register_to_reference(RAS_output,self.atlas_path,"MNI")
        self._print_duration(action_name,time)
        if self.gui != None and self.gui.check_stop():
            raise InterruptedError("Action was cancelled by the user.")

    
        data, spacing, affine, original_shape = self._load_img(MNI_output)
        self._print_shape(data)
        if bbox is None:
            data, seg, bbox = self._crop_to_nonzero(data)
        else:
            data, seg, bbox = self._crop_to_nonzero(data,bbox=bbox)
        self._print_shape(data)

        action_name="resampling"
        new_spacing = (1.0, 1.0, 1.0)
        self._print_action(action_name)
        data, time = self.resampler.run(data,spacing,new_spacing)
        self._print_duration(action_name,time)
        self._print_shape(data)
        if self.gui != None and self.gui.check_stop():
            raise InterruptedError("Action was cancelled by the user.")
        
        action_name="Z score name"
        self._print_action(action_name)
        data, time = self._z_score_norm(data,seg)
        self._print_duration(action_name,time)
        # bbox = tuple(slice(start, end) for start, end in bbox)
        self._print_shape(data)
        if self.gui != None and self.gui.check_stop():
            raise InterruptedError("Action was cancelled by the user.")
        
        action_name="padding"
        self._print_action(action_name)
        data,padding,time = self._padding(data)
        self._print_duration(action_name,time)
        self._print_shape(data)
        return data, affine, bbox, original_shape, trsf_path, spacing, padding
    
    def clean(self,temp_dir):
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                self.logger.info(f"Temporary directory '{temp_dir}' has been removed.")
            except Exception as e:
                self.logger.error(f"Failed to delete temp directory '{temp_dir}': {e}")
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
    def find_nii_files(self):
        nii_paths = {}
        path_dict = {}
        subject_number = 0
        flair_number = 0
        none_list = []    
        input_path=self.option.get("input_path")
        if os.path.isfile(input_path) and input_path.endswith(EXTENSIONS) :
            nii_paths[input_path]=None
            subject_number = 1
            self.option.set("is_file",True)
        elif os.path.isdir(input_path):
            self.option.set("is_file",False)
            rawdata_path = os.path.join(input_path,RAWDATA)
            derivatives_path = os.path.join(input_path,DERIVATIVES)
            basepaths =[]
            if os.path.isdir(rawdata_path):
                basepath.append(rawdata_path)
                if os.path.isdir(derivatives_path):
                    basepaths.append(derivatives_path)
            else:
                basepaths = [input_path]
            for basepath in basepaths:
                for root, _, files in os.walk(basepath):
                    for f in files:
                        if f.endswith(EXTENSIONS):
                            if DERIVATIVES in root and BET not in f:
                                continue
                            f_id = self._get_image_basename(f)
                            if BET in f:
                                f_id = self._rm_entity(f_id,BET)
                                path_dict.setdefault(f_id,{})[BET]=os.path.join(root, f)
                            else : 
                                path_dict.setdefault(f_id,{})["RAW"]=os.path.join(root, f)                        
            subject = {}
            for name,files in path_dict.items():
                if BET in files:
                    f = files[BET]
                else : 
                    f = files["RAW"]
                if self.option.get("flair"):
                    if T1 in name:
                        subject_id = self._rm_entity(name,T1)
                        subject.setdefault(subject_id,{})["T1"]=f
                    elif FLAIR in name:
                        subject_id = self._rm_entity(name,FLAIR)
                        subject.setdefault(subject_id,{})['FLAIR']=f
                else : 
                    nii_paths[f]=None
                    subject_number+=1
            if self.option.get("flair"):
                for subject_id,modalities in subject.items():
                    if "T1" in modalities and "FLAIR" in modalities:
                        nii_paths[modalities["T1"]] = modalities["FLAIR"]
                        flair_number +=1
                        subject_number +=1
                    else :
                        none_list.append(subject_id)
                        if "T1" in modalities:
                            subject_number+=1
                        else:
                            flair_number+=1
        self.logger.debug(f'list : {path_dict}')
        self.logger.debug(f'dict : {nii_paths}')
        return nii_paths,subject_number,flair_number,none_list