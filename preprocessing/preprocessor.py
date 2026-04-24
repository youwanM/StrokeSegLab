import logging
from managers.config_manager import Config
from utils.naming import BET, DERIVATIVES, EXTENSIONS, MNI, RAWDATA, T1
from managers.option_manager import Option
from preprocessing.brain_extraction import BrainExtracter
from preprocessing.resampling import Resampler
from utils.wrapper import AnimaWrapper
from scipy.ndimage import binary_fill_holes
from preprocessing.utils import get_bbox_from_mask, bounding_box_to_slice
from utils.path import ATLAS_DIR

import numpy as np
import nibabel
import os
import shutil
import nibabel as nib

# Removed move_to_output from imports
from utils.processing_utils import get_image_basename, rm_entity

class Preprocessor:
    """
    This class performs preprocessing on 3D T1 images.
    """
    def __init__(self, gui=None) -> None:
        self.logger = logging.getLogger()
        self.option = Option()
        self.config = Config()
        self.resampler = Resampler()
        self.wrapper = AnimaWrapper()
        self.atlasImage = os.path.join(ATLAS_DIR, "Reference_T1.nrrd")
        self.gui = gui
        self.brain_extracter = BrainExtracter(self.wrapper, self.atlasImage, gui)
    
    def _load_img(self, img_path: str) -> tuple[np.ndarray, tuple[float, float, float], np.ndarray, tuple[int, int, int]]:
        img = nibabel.load(img_path)
        affine = img.affine
        spacing = img.header.get_zooms()
        data = img.get_fdata().astype("float32")
        data = np.transpose(data, (2, 1, 0))
        original_shape = data.shape
        if data.ndim == 3:
            data = np.expand_dims(data, axis=0)
        return data, spacing, affine, original_shape
    
    def _z_score_norm(self, data: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        if seg is not None:
            mask = seg >= 0
            mean = data[mask].mean()
            std = data[mask].std()
            data[mask] = (data[mask] - mean) / (max(std, 1e-8))
        else:
            mean = data.mean()
            std = data.std()
            data -= mean
            data /= (max(std, 1e-8))
        return data

    def _bias_correct(self, img_path: str, prefix: str) -> str:
        output_path = prefix + '_N4.nii.gz'
        command = ["animaN4BiasCorrection", "-i", img_path, "-o", output_path]
        self.wrapper.run(command)
        return output_path

    def _register_to_reference(self, img_path: str, ref: str, suffix: str, prefix: str) -> tuple[str, str]:
        output_path = prefix + "_" + suffix + ".nii.gz"
        trsf_path = output_path.replace('.nii.gz', '.txt')
        command = ["animaPyramidalBMRegistration", "-m", img_path, "-r", ref, "-o", output_path, "-O", trsf_path]
        self.wrapper.run(command)
        return output_path, trsf_path

    def _print_action(self, action_name: str) -> None:
        self.logger.info(f"Starting {action_name}...")
        if self.gui != None:
            self.gui.update_status(f"Preprocessing : Starting {action_name}...")

    def _reorient_RAS(self, img_path: str, prefix: str) -> str:
        img = nib.load(img_path)
        reoriented_img = nib.as_closest_canonical(img)
        output_path = prefix + '_RAS.nii.gz'
        nib.save(reoriented_img, output_path)
        return output_path

    def _create_nonzero_mask(self, data: np.ndarray) -> np.ndarray:
        assert data.ndim == 4, "data must have shape (c, x, y, z)"
        nonzero_mask = data[0] != 0
        for c in range(1, data.shape[0]):
            nonzero_mask |= data[c] != 0
        return binary_fill_holes(nonzero_mask)

    def _crop_to_nonzero(self, data: np.ndarray, seg: np.ndarray = None, nonzero_label: int = -1, bbox: tuple[slice, slice, slice] = None) -> tuple[np.ndarray, np.ndarray, tuple[slice, slice, slice]]:
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

    def _padding(self, data: np.ndarray, min_size: int = 128) -> tuple[np.ndarray, list[tuple[int, int]]]:
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
        return padded_data, padding

    def run(self, t1: str, temp_dir: str, bet_only: bool = False) -> tuple:
        self.preprocessing_steps = []
        prefix = os.path.join(temp_dir, get_image_basename(t1))
        
        if not (prefix.endswith((BET, MNI)) or self.option.get("skip_BET")): 
            if self.gui != None and self.gui.check_stop():
                raise InterruptedError("Action was cancelled by the user.")
            action_name = "brain extraction"
            self._print_action(action_name)
            bet_t1 = self.brain_extracter.run(t1, prefix)
            if self.gui != None and self.gui.check_stop():
                raise InterruptedError("Action was cancelled by the user.")
        else:
            bet_t1 = shutil.copy(t1, temp_dir) 

        if not bet_only: 
            is_mni = prefix.endswith(MNI)
            data_t1, affine, bbox, original_shape, trsf_path, spacing, padding, MNI_base_image = self._preprocess_modality(bet_t1, is_mni)
            
            # Removed move_to_output calls entirely. 
            # bet_t1 and all intermediate files now safely remain in the Session temp_dir.
                    
            return data_t1, affine, bbox, original_shape, trsf_path, spacing, padding, bet_t1, MNI_base_image
        else:
            return bet_t1

    def _preprocess_modality(self, modality: str, is_MNI: bool, bbox: tuple[slice, slice, slice] = None) -> tuple:
        if not is_MNI:
            prefix = os.path.join(os.path.dirname(modality), rm_entity(modality, BET))
            
            action_name = "bias correction"
            self._print_action(action_name)
            n4_output = self._bias_correct(modality, prefix)
            self.preprocessing_steps.append(n4_output)
            if self.gui != None and self.gui.check_stop(): raise InterruptedError()

            action_name = "reorient to RAS"
            self._print_action(action_name)
            RAS_output = self._reorient_RAS(n4_output, prefix)
            self.preprocessing_steps.append(RAS_output)
            if self.gui != None and self.gui.check_stop(): raise InterruptedError()
            
            action_name = "register to MNI"
            self._print_action(action_name)
            MNI_output, trsf_path = self._register_to_reference(RAS_output, self.atlasImage, MNI, prefix)
            if self.gui != None and self.gui.check_stop(): raise InterruptedError()
        else:
            MNI_output = modality
            trsf_path = None

        if self.option.get("keep_MNI"):
            if BET in MNI_output:
                new_output = rm_entity(MNI_output, BET) + "_" + MNI + ".nii.gz"
                new_output = os.path.join(os.path.dirname(MNI_output), new_output)
                MNI_output = shutil.copy(MNI_output, new_output)
            # Retain in the temp_dir instead of moving to the input folder
            MNI_base_image = MNI_output
        else:
            self.preprocessing_steps.append(MNI_output)
            MNI_base_image = None
            
        data, spacing, affine, original_shape = self._load_img(MNI_output)
        data, seg, bbox = self._crop_to_nonzero(data, bbox=bbox)
        
        action_name = "resampling"
        self._print_action(action_name)
        data = self.resampler.run(data, spacing, (1.0, 1.0, 1.0))
        if self.gui != None and self.gui.check_stop(): raise InterruptedError()
        
        action_name = "Z score norm"
        self._print_action(action_name)
        data = self._z_score_norm(data, seg)
        if self.gui != None and self.gui.check_stop(): raise InterruptedError()
        
        action_name = "padding"
        self._print_action(action_name)
        data, padding = self._padding(data)
        
        return data, affine, bbox, original_shape, trsf_path, spacing, padding, MNI_base_image
    
    def clean(self, temp_dir):
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                self.logger.info(f"Temporary directory '{temp_dir}' has been removed.")
            except Exception as e:
                self.logger.error(f"Failed to delete temp directory '{temp_dir}': {e}")

    def find_nii_files(self) -> tuple[list[str], int]:
        """
        Search for NIfTI (.nii or .nii.gz) T1 files in the input directory.

        Returns:
            tuple[list[str], int]: 
            - List of T1 file paths
            - The number of T1 subjects found
        """
        nii_paths = [] 
        subject_number = 0
        input_path = self.option.get("input_path")
        
        if os.path.isfile(input_path) and input_path.endswith(EXTENSIONS):
            nii_paths.append(input_path)
            subject_number = 1
            self.option.set("is_file", True) 
        elif os.path.isdir(input_path):
            self.option.set("is_file", False)
            rawdata_path = os.path.join(input_path, RAWDATA)
            derivatives_path = os.path.join(input_path, DERIVATIVES)
            
            basepaths = [rawdata_path, derivatives_path] if os.path.isdir(rawdata_path) else [input_path]
            
            path_dict = {} 
            for basepath in basepaths:
                if not os.path.isdir(basepath): continue
                for root, _, files in os.walk(basepath):
                    for f in files:
                        if f.endswith(EXTENSIONS):
                            if DERIVATIVES in root and BET not in f and not (self.option.get("keep_MNI", False) and MNI in f):
                                continue
                            f_id = get_image_basename(f)
                            if BET in f:
                                f_id = rm_entity(f_id, BET)
                                path_dict.setdefault(f_id, {})[BET] = os.path.join(root, f)
                            elif MNI in f:
                                f_id = rm_entity(f_id, MNI)
                                path_dict.setdefault(f_id, {})[MNI] = os.path.join(root, f)
                            else: 
                                path_dict.setdefault(f_id, {})["RAW"] = os.path.join(root, f)                        
            
            for name, files in path_dict.items():
                f = files.get(MNI) or files.get(BET) or files.get("RAW")
                if T1 in name: 
                    nii_paths.append(f)
                    subject_number += 1
                    
        return nii_paths, subject_number