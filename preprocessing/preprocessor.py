import logging
from utils.config_manager import Config
from utils.naming import BET, DERIVATIVES, EXTENSIONS, FLAIR, MNI, RAWDATA, T1
from utils.option_manager import Option
from preprocessing.brain_extraction import BrainExtracter
from preprocessing.resampling import Resampler
from preprocessing.wrapper import AnimaWrapper
from scipy.ndimage import binary_fill_holes
from preprocessing.utils import get_bbox_from_mask, bounding_box_to_slice
from utils.path import ATLAS_DIR

import numpy as np
import nibabel
import os
import shutil
import nibabel as nib

from utils.processing_utils import get_image_basename, move_to_output, rm_entity

class Preprocessor:
    """
    This class performs preprocessing on 3D images
    """
    def __init__(self,gui =None)-> None:
        """
        Initialize the main preprocessing class with optional GUI integration.

        Args:
            gui (GUIMain, optional): Instance of the gui class to enable the display of messages and status updates. Defaults to None.
        """
        self.logger = logging.getLogger()
        self.option = Option()
        self.config = Config()
        self.resampler = Resampler()
        self.wrapper = AnimaWrapper()
        self.atlasImage = os.path.join(ATLAS_DIR,"Reference_T1.nrrd")
        self.gui = gui
        self.brain_extracter = BrainExtracter(self.wrapper,self.atlasImage,gui)
    
    def _load_img(self,img_path : str)->tuple[np.ndarray, tuple[float,float,float], np.ndarray, tuple[int,int,int]]:
        """
        Load a NIFTI image from the given file path.
        - Load using nibabel
        - extract affine matrix and voxel spacing
        - extract, transpose and add a channel to data if necessary : (z, y, x) -> (c, x, y, z)
        - Store original shape (used later in postprocessing)

        Args:
            img_path (str): The image path

        Returns:
            tuple[np.ndarray, tuple[float,float,float], np.ndarray, tuple[int,int,int]]: Data loaded, spacing, affine matrix and the original shape
        """
        img = nibabel.load(img_path)
        affine = img.affine
        spacing = img.header.get_zooms()
        data = img.get_fdata().astype("float32")
        data = np.transpose(data, (2, 1, 0))
        original_shape=data.shape
        if data.ndim==3:
            data = np.expand_dims(data,axis=0)
        return data, spacing, affine, original_shape
    
    def _z_score_norm(self,data : np.ndarray, seg : np.ndarray=None) -> np.ndarray:
        """
        Apply z-score normalization to the input data. If a segmentation mask is provided, the normalization is applied only on the region where the mask is non-negative. Otherwise, the entire volume is normalized.

        Args:
            data (np.ndarray): Input data (c, x, y, z)
            seg (np.ndarray, optional): _description_. Defaults to None.

        Returns:
            np.ndarray: _description_
        """

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
        return data

    def _bias_correct(self,img_path : str)-> str:
        """
        Use an Anima executable through a wrapper to perform bias correction on the image

        Args:
            img_path (str): Path of the image

        Returns:
            str: Output path
        """
        output_path=img_path.replace('.nii.gz', '_N4.nii.gz')
        command=["animaN4BiasCorrection","-i",img_path,"-o",output_path]
        self.wrapper.run(command)
        return output_path

        
    def _register_to_reference(self,img_path : str ,ref : str ,suffix : str ,prefix : str =None) -> tuple[str,str]:
        """
        Register an image to a reference using Anima's pyramidal block matching registration.

        Args:
            img_path (str): Image path
            ref (str): Reference image path
            suffix (str): Suffix to add to the output registered image filename
            prefix (str, optional): Custom prefix for the output image filename. Allows to separate different registration. Defaults to None.

        Returns:
            tuple[str,str]: Output path of the registered image, Output path of the transformation file (.txt)
        """
        if prefix is None:
            if img_path.endswith(".nii.gz"):
                output_path= img_path.replace('.nii.gz', "_"+suffix+'.nii.gz')
            elif img_path.endswith(".nii"):
                output_path= img_path.replace('.nii', "_"+suffix+'.nii.gz') # needs to handle .nii.gz and .nii file too because it's the first step of preprocessing 
        else : 
            output_path = prefix + "_" + suffix + ".nii.gz"
        trsf_path = output_path.replace('.nii.gz', '.txt')
        command=["animaPyramidalBMRegistration","-m",img_path,"-r",ref,"-o",output_path,"-O",trsf_path]
        self.wrapper.run(command)
        return output_path,trsf_path

    def _print_action(self,action_name : str)->None:
        """
        Log the current action
        Args:
            action_name (str): Name of the action
        """
        self.logger.info(f"Starting {action_name}...")
        if(self.gui !=None):
            self.gui.update_status(f"Preprocessing : Starting {action_name}...")

    def _reorient_RAS(self,img_path : str)->str:
        """
        Reorient to RAS an image

        Args:
            img_path (str): Image path

        Returns:
            str: Output path
        """
        img = nib.load(img_path)
        reoriented_img = nib.as_closest_canonical(img)
        output_path=img_path.replace('.nii.gz','_RAS.nii.gz')
        nib.save(reoriented_img, output_path)
        return output_path


    def _create_nonzero_mask(self,data : np.ndarray)->np.ndarray:
        """
        Create a binary mask indicating where the input data is non-zero

        Args:
            data (np.ndarray): Input array of shape (c, x, y, z)

        Returns:
            np.ndarray: Binary mask (x, y, z) with True where any channel is non-zero.
        """
        assert data.ndim ==4, "data must have shape (c, x, y, z)"
        nonzero_mask = data[0] != 0
        for c in range(1, data.shape[0]):
            nonzero_mask |= data[c] != 0
        return binary_fill_holes(nonzero_mask)


    def _crop_to_nonzero(self,data : np.ndarray, seg : np.ndarray =None, nonzero_label : int =-1,bbox : tuple[slice, slice, slice] =None)-> tuple[np.ndarray, np.ndarray, tuple[slice, slice, slice]]:
        """
        Crop the input data and segmentation to the non-zero region. Optionally reuses a provided bounding box (Used to crop the FLAIR image using the same bounding box as the T1 image)

        Args:
            data (np.ndarray): Input array of shape (c, x, y, z)
            seg (np.ndarray, optional): Corresponding segmentation array. Defaults to None.
            nonzero_label (int, optional): Label used to mark regions outside the non-zero crop. Defaults to -1.
            bbox (tuple[slice, slice, slice], optional): _description_. Defaults to None.

        Returns:
            tuple[np.ndarray, np.ndarray, tuple[slice, slice, slice]]: Cropped image data, Cropped segmentation mask, Bounding box used for cropping
        """
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

    def _padding(self,data : np.ndarray, min_size : int =128)->tuple[np.ndarray, list[tuple[int, int]]]:
        """
        Pad the data with zeros so that each dimension is at least min_size
        If the input data has 4 dimensions and the first dimension (channels) is 1, this dimension is squeezed out before padding and then restored afterwards
        The padding is applied symmetrically (half before, half after) on each dimension.

        Args:
            data (np.ndarray): Input array of shape (c, x, y, z)
            min_size (int, optional): Minimum size. Defaults to 128.

        Returns:
            tuple[np.ndarray, list[tuple[int, int]]]: Padded data, List of tuples indicating how much padding was added before and after each dimension
        """
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

    def run(self,t1 : str, flair : str, temp_dir :str ,bet_only : bool =False) -> tuple[np.ndarray,np.ndarray,list[tuple[int, int]], tuple[int,int,int], str, tuple[float,float,float], list[tuple[int, int]], str, str]:
        """
        Run the brain extraction and handle different case : 1 or 2 channels, BET only or not, MNI input or not, save MNI or not

        Args:
            t1 (str): Path to the T1 image
            flair (str): Path to the FLAIR image
            temp_dir (str): Path to the temporary directory
            bet_only (bool, optional): If True doing brain extraction only. Defaults to False.

        Returns:
            tuple[np.ndarray,np.ndarray,list[tuple[int, int]], tuple[int,int,int], str, tuple[float,float,float], list[tuple[int, int]], str, str]: 
            - data preprocessed 
            - affine matrix 
            - Bounding box coordinates used for cropping
            - Original shape of the image before preprocessing
            - Path to the transformation text file
            - Voxel spacing of the image
            - List of padding tuples applied to each dimension
            - Path to the brain-extracted T1 image
            - Reference MNI image array or None
        """
        prefix = os.path.join(temp_dir,get_image_basename(t1))
        if not prefix.endswith((BET,MNI)): 
            if self.gui != None and self.gui.check_stop():
                raise InterruptedError("Action was cancelled by the user.")
            action_name="brain extraction"
            self._print_action(action_name)
            bet_t1 = self.brain_extracter.run(t1,prefix)
            if self.gui != None and self.gui.check_stop():
                raise InterruptedError("Action was cancelled by the user.")
        else :
            bet_t1 = shutil.copy(t1,temp_dir) # If prefix ends with BET or MNI, the previous steps have already been done. We just copy the T1 in the tempory directory
        if not bet_only: 
            if prefix.endswith(MNI):
                data_t1, affine, bbox, original_shape, trsf_path, spacing, padding, MNI_base_image = self._preprocess_modality(bet_t1,True)
            else:
                data_t1, affine, bbox, original_shape, trsf_path, spacing, padding, MNI_base_image = self._preprocess_modality(bet_t1,False)
                move_to_output(bet_t1) # Save the brain extracted image
        else:
            move_to_output(bet_t1) # if it's brain extraction only, just need to call move_to_output with the bet image

        if self.option.get("flair"): # If the model is a T1/FLAIR, apply almost the same steps to the FLAIR image as done for the T1 image ()
            if flair is not None:
                prefix = os.path.join(temp_dir,get_image_basename(flair))
                if not prefix.endswith((BET,MNI)):
                    if self.gui != None and self.gui.check_stop():
                        raise InterruptedError("Action was cancelled by the user.")
                    action_name = "register FLAIR to T1" # One more step before: register the FLAIR image to the T1, so it can be preprocessed the same way
                    self._print_action(action_name)
                    flair_t1,_ = self._register_to_reference(flair, t1,T1,prefix=prefix)
                    if self.gui != None and self.gui.check_stop():
                        raise InterruptedError("Action was cancelled by the user.")
                    action_name="brain extraction"
                    self._print_action(action_name)
                    bet_flair= self.brain_extracter.run(flair_t1,prefix)
                    if self.gui != None and self.gui.check_stop():
                        raise InterruptedError("Action was cancelled by the user.")
                else :
                    bet_flair = shutil.copy(flair,temp_dir)
                if not bet_only:
                    if prefix.endswith(MNI):
                        data_flair, *_ = self._preprocess_modality(bet_flair,True,bbox=bbox)
                    else : 
                        data_flair, *_ = self._preprocess_modality(bet_flair,False,bbox=bbox)
                        move_to_output(bet_flair)
                    data = np.concatenate([data_t1, data_flair], axis=0) # The model expects a NumPy array with two channels, so we concatenate along the channel axis
                    return data, affine, bbox, original_shape, trsf_path, spacing, padding, bet_t1, MNI_base_image
                else : 
                    move_to_output(bet_flair)
            else:
                raise ValueError("FLAIR image is required but was not provided") # Normally, it doesn't happen because find_nii_files return only subjects with both a T1 and a FLAIR
        elif not bet_only:
            return data_t1, affine, bbox, original_shape, trsf_path, spacing, padding, bet_t1, MNI_base_image

    
    def _preprocess_modality(self,modality : str ,is_MNI : bool ,bbox : tuple[slice, slice, slice] = None)-> tuple[np.ndarray,np.ndarray,list[tuple[int, int]], tuple[int,int,int], str, tuple[float,float,float], list[tuple[int, int]], str]:
        """
        Run the full preprocessing pipeline except brain extraction : 
        - If the image is not already register to MNI : bias correction, reorient to RAS, register to MNI
        - Resampling, Z score, padding

        Args:
            modality (str): Image path
            is_MNI (bool): True if the image is already register to MNI
            bbox (tuple[slice, slice, slice], optional): Bounding box coordinates used for cropping. Defaults to None.

        Returns:
            tuple[np.ndarray,np.ndarray,list[tuple[int, int]], tuple[int,int,int], str, tuple[float,float,float], list[tuple[int, int]], str]:
            - data preprocessed 
            - affine matrix 
            - Bounding box coordinates used for cropping
            - Original shape of the image before preprocessing
            - Path to the transformation text file
            - Voxel spacing of the image
            - List of padding tuples applied to each dimension
            - Reference MNI image array or None    
        """
        if not is_MNI:
            action_name="bias correction"
            self._print_action(action_name)
            n4_output=self._bias_correct(modality)
            if self.gui != None and self.gui.check_stop():
                raise InterruptedError("Action was cancelled by the user.")

            action_name="reorient to RAS"
            self._print_action(action_name)
            RAS_output=self._reorient_RAS(n4_output)
            if self.gui != None and self.gui.check_stop():
                raise InterruptedError("Action was cancelled by the user.")
            
            action_name="register to MNI"
            self._print_action(action_name)
            MNI_output,trsf_path=self._register_to_reference(RAS_output,self.atlasImage,MNI)
            if self.gui != None and self.gui.check_stop():
                raise InterruptedError("Action was cancelled by the user.")
        else : 
            MNI_output = modality
            trsf_path = None

        if self.option.get("keep_MNI"):
            if BET in MNI_output:
                new_output = rm_entity(MNI_output,BET) + "_" + MNI + ".nii.gz"
                new_output = os.path.join(os.path.dirname(MNI_output),new_output)
                MNI_output=shutil.copy(MNI_output,new_output)
            MNI_base_image = move_to_output(MNI_output)
        else:
            MNI_base_image = None
        data, spacing, affine, original_shape = self._load_img(MNI_output)
        if bbox is None:
            data, seg, bbox = self._crop_to_nonzero(data)
        else:
            data, seg, bbox = self._crop_to_nonzero(data,bbox=bbox)
        
        action_name="resampling"
        new_spacing = (1.0, 1.0, 1.0)
        self._print_action(action_name)
        data= self.resampler.run(data,spacing,new_spacing)
        if self.gui != None and self.gui.check_stop():
            raise InterruptedError("Action was cancelled by the user.")
        
        action_name="Z score norm"
        self._print_action(action_name)
        data= self._z_score_norm(data,seg)
        if self.gui != None and self.gui.check_stop():
            raise InterruptedError("Action was cancelled by the user.")
        
        action_name="padding"
        self._print_action(action_name)
        data,padding= self._padding(data)
        return data, affine, bbox, original_shape, trsf_path, spacing, padding, MNI_base_image
    
    def clean(self,temp_dir):
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                self.logger.info(f"Temporary directory '{temp_dir}' has been removed.")
            except Exception as e:
                self.logger.error(f"Failed to delete temp directory '{temp_dir}': {e}")

    def find_nii_files(self) -> tuple[dict[str, str | None], int, int, list[str]]:
        """
        Search for NIfTI (.nii or .nii.gz) files in the input directory or single input file.

        Returns:
            tuple[dict[str, str | None], int, int, list[str]]: 
            - Dictionary : T1 path : FLAIR path or None
            - The number of T1 subjects (FLAIR too if using a 1 channel model)
            - The number of FLAIR images (if using a 2 channels model)
            - List of subject IDs for which T1 or FLAIR was missing
        """
        nii_paths = {} # final dict
        path_dict = {} # intermediate dict : {"sub1T1" : {"RAW" : "path/to/raw", "BET" : "path/to/bet", "MNI" : "path/to/mni"}, "sub1FLAIR" : {...}}
        subject_number = 0
        flair_number = 0
        none_list = [] # final list of subject with a T1 or FLAIR missing (2 channels model)
        input_path=self.option.get("input_path")
        if os.path.isfile(input_path) and input_path.endswith(EXTENSIONS) :
            nii_paths[input_path]=None # if the input path is a file, we just put the path in the dict
            subject_number = 1
            self.option.set("is_file",True) 
        elif os.path.isdir(input_path):
            self.option.set("is_file",False)
            rawdata_path = os.path.join(input_path,RAWDATA)
            derivatives_path = os.path.join(input_path,DERIVATIVES)
            basepaths =[]
            if os.path.isdir(rawdata_path): # Checking if a rawdata and a derivatives path exist (BIDS)
                basepaths.append(rawdata_path)
                if os.path.isdir(derivatives_path):
                    basepaths.append(derivatives_path)
            else:
                basepaths = [input_path]
            for basepath in basepaths:
                for root, _, files in os.walk(basepath):
                    for f in files:
                        if f.endswith(EXTENSIONS):
                            if DERIVATIVES in root and BET not in f and not (self.option.get("keep_MNI",False) and MNI in f): # If the file is in the derivatives folder and is not a BET or MNI with the keep MNI option True, we don't keep the file (avoid segmentation files)
                            # If it's a MNI but the option is False, we won't be able to register the output in the patient space without the trsf_path, so we don't keep it
                                continue
                            f_id = get_image_basename(f)
                            if BET in f:
                                f_id = rm_entity(f_id,BET)
                                path_dict.setdefault(f_id,{})[BET]=os.path.join(root, f) # If f_id not in path_dict, create an empty dict for it. Then, store the path to the BET file
                            elif MNI in f:
                                f_id = rm_entity(f_id,MNI)
                                path_dict.setdefault(f_id,{})[MNI]=os.path.join(root, f)
                            else : 
                                path_dict.setdefault(f_id,{})["RAW"]=os.path.join(root, f)                        
            subject = {} # intermediate dict : {"sub1": {"T1": "path/to/t1", "FLAIR": "path/to/flair"}}
            for name,files in path_dict.items():
                if MNI in files:
                    f = files[MNI]
                elif BET in files:
                    f = files[BET]
                else : 
                    f = files["RAW"]
                if self.option.get("flair"):
                    if T1 in name:
                        subject_id = rm_entity(name,T1)
                        subject.setdefault(subject_id,{})["T1"]=f
                    elif FLAIR in name:
                        subject_id = rm_entity(name,FLAIR)
                        subject.setdefault(subject_id,{})['FLAIR']=f
                elif T1 in name : # if not in 2 channels mode and if the file is a T1, we just fill the final dict
                    nii_paths[f]=None
                    subject_number+=1
            if self.option.get("flair"):
                for subject_id,modalities in subject.items():
                    if "T1" in modalities and "FLAIR" in modalities: # we only keep subject with both a T1 and a FLAIR
                        nii_paths[modalities["T1"]] = modalities["FLAIR"]
                        flair_number +=1
                        subject_number +=1
                    else :
                        none_list.append(subject_id)
                        if "T1" in modalities:
                            subject_number+=1
                        else:
                            flair_number+=1
        return nii_paths,subject_number,flair_number,none_list