import logging
import shutil
import nibabel
import os
import numpy as np
from scipy.special import softmax
from managers.config_manager import Config
from utils.naming import BET, DERIVATIVES, EXTENSIONS, MNI, PMAP, RAWDATA, T1
from managers.option_manager import Option
from postprocessing.viewer import Viewer
from preprocessing.resampling import Resampler
from utils.wrapper import AnimaWrapper
from utils.processing_utils import get_image_basename, move_to_output, rm_entity, create_disclaimer_if_missing

class Postprocessor:
    """
    This class performs postprocessing on 3D images
    """
    def __init__(self, gui=None) -> None:
        self.option = Option()
        self.logger = logging.getLogger()
        self.wrapper = AnimaWrapper()
        self.resampler = Resampler()
        self.config = Config()
        self.viewer = Viewer()
        self.gui = gui
    
    def _save_img(self, temp_dir: str, data: np.ndarray, base_name: str, affine: np.ndarray, name: str) -> str:
        out_img = nibabel.Nifti1Image(data, affine)
        if name == "pmap":
            output_file = os.path.join(temp_dir, base_name + f"_{PMAP}.nii.gz")
        else:
            suffix = self.config.get("default", "suffix")
            output_file = os.path.join(temp_dir, base_name + f"_{suffix}.nii.gz")
        nibabel.save(out_img, output_file)
        return output_file
    
    def _convert_to_segmentation(self, data: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
        data = data[0]
        data = softmax(data, axis=0)
        data = data[1]
        
        pmap = data if self.option.get("save_pmap") else None
        seg = (data >= threshold).astype(np.uint8)
        return seg, pmap
    
    def _register_seg_to_reference(self, img_path: str, trsf_path: str, ref: str) -> None:
        xml_path = trsf_path.replace('.txt', '.xml')
        command = ["animaTransformSerieXmlGenerator", "-i", trsf_path, "-o", xml_path]
        self.wrapper.run(command)

        command = ["animaApplyTransformSerie", "-i", img_path, "-t", xml_path, "-o", img_path, "-g", ref, "-I"]
        self.wrapper.run(command)
        self._binarize_seg(img_path)

    def _binarize_seg(self, seg_path: str, threshold: float = 0.5) -> None:
        img = nibabel.load(seg_path)
        data = img.get_fdata()
        binary = (data >= threshold).astype(np.uint8)
        nibabel.save(nibabel.Nifti1Image(binary, img.affine, img.header), seg_path)

    def _print_action(self, action_name: str) -> None:
        self.logger.info(f"Starting {action_name}...")
        if self.gui != None:
            self.gui.update_status(f"Postprocessing : Starting {action_name}...")
    
    def _remove_padding(self, data: np.ndarray, padding: list[tuple[int, int]]) -> np.ndarray:
        slices = []
        for dim_pad in padding:
            start = dim_pad[0]
            end = -dim_pad[1] if dim_pad[1] > 0 else None
            slices.append(slice(start, end))
        return data[tuple(slices)]

    def _uncrop_from_bbox(self, data: np.ndarray, slicer: tuple[slice, slice, slice], original_shape: tuple[int, int, int]) -> np.ndarray:
        full_volume = np.zeros(original_shape, dtype=data.dtype)
        full_volume[slicer] = data
        full_volume = np.transpose(full_volume, (2, 1, 0))
        return full_volume
    
    def check_viewer(self, viewer: str) -> None:
        self.viewer.check_viewer(viewer)

    def run(self, data, affine, input_path, bbox, original_shape, temp_dir, trsf_path, old_spacing, padding, bet, MNI_base_image, threshold, open_viewer=False, output_dir=None) -> None:
        action_name = "convert to segmentation"
        self._print_action(action_name)
        seg, pmap = self._convert_to_segmentation(data, threshold)

        outputs = [("seg", seg)]
        if pmap is not None:
            outputs.append(("pmap", pmap))
        
        for name, output in outputs:
            action_name = "remove padding"
            self._print_action(action_name)
            output = self._remove_padding(output, padding)

            action_name = "uncrop"
            self._print_action(action_name)
            slicer = tuple(slice(start, end) for start, end in bbox)
            output = self._uncrop_from_bbox(output, slicer, original_shape)

            action_name = "resampling"
            new_spacing = (1.0, 1.0, 1.0)
            self._print_action(action_name)
            output = np.expand_dims(output, axis=0)
            output = self.resampler.run(output, new_spacing, old_spacing)
            output = output.squeeze(0)

            basename = rm_entity(input_path, BET)

            action_name = "saving image to nii"
            self._print_action(action_name)
            nii_file = self._save_img(temp_dir, output, basename, affine, name)

            if trsf_path is None or self.option.get("keep_MNI"):
                new_output = get_image_basename(nii_file) + "_" + MNI + ".nii.gz"
                new_output = os.path.join(os.path.dirname(nii_file), new_output)
                nii_file = shutil.copy(nii_file, new_output)
                input_path = MNI_base_image
            else:
                action_name = "register to reference"
                self._print_action(action_name)
                self._register_seg_to_reference(nii_file, trsf_path, bet)

            is_mask = (name == "seg")
            output_path = move_to_output(nii_file, is_clinical_result=is_mask)
            
            if open_viewer and name == "seg":
                action_name = "open viewer"
                self._print_action(action_name)
                self.viewer.run(input_path, output_path)