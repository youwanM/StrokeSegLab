import logging
import shutil
import nibabel
import os
import numpy as np
from scipy.special import expit, softmax
from managers.config_manager import Config
from utils.naming import BET, DERIVATIVES, EXTENSIONS, MNI, PMAP, RAWDATA, T1
from managers.option_manager import Option
from postprocessing.viewer import Viewer
from preprocessing.resampling import Resampler
from utils.wrapper import AnimaWrapper
from utils.processing_utils import get_image_basename, move_to_output, rm_entity

class Postprocessor:
    """
    This class performs postprocessing on 3D images
    """
    def __init__(self,gui=None)-> None:
        """
        Initialize the main postprocessing class with optional GUI integration.

        Args:
            gui (GUIMain, optional): Instance of the gui class to enable the display of messages and status updates. Defaults to None.
        """
        self.option = Option()
        self.logger =logging.getLogger()
        self.wrapper = AnimaWrapper()
        self.resampler = Resampler()
        self.config = Config()
        self.viewer = Viewer()
        self.gui = gui
    
    def _save_img(self,temp_dir :str ,data : np.ndarray ,base_name : str,affine : np.ndarray, name : str)->str:
        """
        Save a NIFTI image with a dynamically constructed name

        Args:
            temp_dir (str): Directory where the image will be save
            data (np.ndarray): Data of the image 
            base_name (str): Basename of the image, without extension
            affine (np.ndarray): Affine transformation matrix of the image
            name (str): Identifier to determine filename suffix

        Returns:
            str: Full path of the saved file
        """
        out_img = nibabel.Nifti1Image(data,affine)
        if name =="pmap":
            output_file = os.path.join(temp_dir, base_name + f"_{PMAP}.nii.gz")
        else:
            suffix = self.config.get("default","suffix")
            output_file = os.path.join(temp_dir, base_name + f"_{suffix}.nii.gz")
        nibabel.save(out_img, output_file)
        return output_file
    
    def _convert_to_segmentation(self, data : np.ndarray,threshold: float)->tuple[np.ndarray,np.ndarray]:
        """
        Convert the output of the model into a segmentation based on the threshold given : 
        - Take the first element of the batch axis
        - Apply a softmax on the channel dimensions (2 channels : background and lesion)
        - Exctract the probabilities of the lesion class (1)
        - Save it if the option 'pmap' is specified
        - Apply a threshold to create a binary mask

        Args:
            data (np.ndarray): Model output with shape (batch_size, channels, x, y, z)
            threshold (float): Threshold to create the binary mask

        Returns:
            tuple[np.ndarray,np.ndarray]: seg : binary mask, pmap : probability map if requested
        """
        data = data[0]
        # data = data[1]
        # data = expit(data)
        data = softmax(data,axis=0)
        data = data[1]
        if self.option.get("save_pmap"):
            pmap = data
        else:
            pmap = None
        seg = (data >= threshold).astype(np.uint8)
        return seg,pmap
    
    def _register_to_reference(self,img_path : str,trsf_path : str,ref : str)-> None:
        """
        Register an image to a reference using anima executable and a wrapper

        Args:
            img_path (str): Image path
            trsf_path (str): Path to the transformation text file, produced during the preprocessing
            ref (str): Reference image path
        """
        xml_path = trsf_path.replace('.txt','.xml')
        command=["animaTransformSerieXmlGenerator","-i",trsf_path,"-o",xml_path]
        self.wrapper.run(command)

        command=["animaApplyTransformSerie","-i",img_path,"-t",xml_path,"-o",img_path,"-g",ref,"-I"]
        self.wrapper.run(command)

    def _print_action(self,action_name : str)-> None:
        """
        Log the current action
        Args:
            action_name (str): Name of the action
        """
        self.logger.info(f"Starting {action_name}...")
        if(self.gui !=None):
            self.gui.update_status(f"Postprocessing : Starting {action_name}...")
    
    def _remove_padding(self,data : np.ndarray, padding : list[tuple[int, int]])->np.ndarray:
        """
        Remove padding from a numpy array based on a list of tuples specifying the intervals to keep for each dimension

        Args:
            data (np.ndarray): Data numpy array to process
            padding (list[tuple[int, int]]): List of tuples that specify the padding to remove (start, end)

        Returns:
            np.ndarray: Array with padding removed
        """
        slices = []
        for dim_pad in padding:
            start = dim_pad[0]
            end = -dim_pad[1] if dim_pad[1] > 0 else None
            slices.append(slice(start, end))
        return data[tuple(slices)]

    def _uncrop_from_bbox(self,data : np.ndarray,slicer : tuple[slice,slice,slice],original_shape : tuple[int,int,int])->np.ndarray:
        """
        Place the cropped data back into a full-size volume at the location specified by slicer. Then transpose the volume axes from (x, y, z) to (z, y, x)

        Args:
            data (np.ndarray): Cropped data
            slicer (tuple[slice,slice,slice]): Tuple of slices that specify where to place data
            original_shape (tuple[int,int,int]): Shape of full original volume

        Returns:
            np.ndarray: Uncropped data
        """
        full_volume = np.zeros(original_shape, dtype=data.dtype)
        full_volume[slicer]=data
        full_volume = np.transpose(full_volume, (2, 1, 0))
        return full_volume
    
    def check_viewer(self, viewer : str) -> None:
        """
        Call for check_viewer method from the viewer class to check if the viewer specify exist

        Args:
            viewer (str): Name of the viewer specify
        """
        self.viewer.check_viewer(viewer)


    
    def run(self,data : np.ndarray ,affine : np.ndarray ,input_path : str ,bbox : list[tuple[int, int]] ,original_shape : tuple[int,int,int],temp_dir : str ,trsf_path : str,old_spacing : tuple[float,float,float],padding : list[tuple[int, int]],bet : str ,MNI_base_image : str,threshold : float ,open_viewer : bool =False) -> None:
        """
        Run the entire postprocessing pipeline on the data produced by the inference step : 
        - Convert to segmentation, return a pmap if the option is specified by the user
        - Remove padding
        - Uncrop
        - Resampling to the original spacing
        - Saving image
        - Register to reference only if the inverse transformation was applied during preprocessing
        - Open viewer with input image and binary mask if the option was specified by the user


        Args:
            data (np.ndarray): Data prostprocessed
            affine (np.ndarray): affine matrix
            input_path (str): Input path
            bbox (list[tuple[int, int]]): Bounding box coordinates used for cropping
            original_shape (tuple[int,int,int]): Original shape of the image before preprocessing
            temp_dir (str): Path to the temporary directory
            trsf_path (str): Path to the transformation text file
            old_spacing (tuple[float,float,float]): Voxel spacing of the input image
            padding (list[tuple[int, int]]): List of padding tuples applied to each dimension
            bet (str): Path to the brain-extracted T1 image
            MNI_base_image (str): Reference MNI image array or None
            threshold (float): Segmentation threshold value
            open_viewer (bool, optional): If True open a viewer with input image and segmentation. Defaults to False.
        """


        action_name="convert to segmentation"
        self._print_action(action_name)
        seg,pmap = self._convert_to_segmentation(data,threshold)

        outputs = [("seg",seg)]
        if pmap is not None:
            outputs.append(("pmap",pmap))
        # List of (name, array) tuples used to postprocess both binary mask and probability map differently depending on their type
        
        for name,output in outputs:

            action_name="remove padding"
            self._print_action(action_name)
            output = self._remove_padding(output,padding)

            action_name="uncrop"
            self._print_action(action_name)
            slicer = tuple(slice(start, end) for start, end in bbox)
            output = self._uncrop_from_bbox(output,slicer,original_shape)

            action_name="resampling"
            new_spacing = (1.0, 1.0, 1.0)
            self._print_action(action_name)
            output = np.expand_dims(output, axis=0)
            output= self.resampler.run(output,new_spacing,old_spacing)
            output = output.squeeze(0)

            basename = rm_entity(input_path,BET)
            if self.option.get("flair"):
                basename = rm_entity(basename,T1)

            action_name="saving image to nii"
            self._print_action(action_name)
            nii_file = self._save_img(temp_dir,output,basename,affine,name)

            if trsf_path is None or self.option.get("keep_MNI"):
                new_output = get_image_basename(nii_file) + "_" + MNI + ".nii.gz"
                new_output = os.path.join(os.path.dirname(nii_file),new_output)
                nii_file = shutil.copy(nii_file,new_output)
                input_path = MNI_base_image
            else:
                action_name="register to reference"
                self._print_action(action_name)
                self._register_to_reference(nii_file,trsf_path,bet)

            output_path = move_to_output(nii_file)
            if open_viewer and name=="seg":
                action_name="open viewer"
                self._print_action(action_name)
                self.viewer.run(input_path,output_path)