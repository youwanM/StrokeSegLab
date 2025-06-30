import logging
import nibabel
import os
import numpy as np

from manager.option_manager import Option

class Postprocessor:
    def __init__(self):
        self.option = Option()
        self.logger =logging.getLogger()
        
    
    def _save_img(self,output_path,data,input_path,affine):
        out_img = nibabel.Nifti1Image(data,affine)
        base_name = os.path.basename(input_path)
        base_name = base_name.split(".nii")[0]
        output_file = os.path.join(output_path, base_name + "_seg.nii.gz")
        self.logger.info(f"Saving segmented image to: {output_file}")
        nibabel.save(out_img, output_file)
    
    def _convert_to_segmentation(self, data):
        data = data[0]
        seg = np.argmax(data, axis=0).astype(np.uint8)
        return seg
    
    def run(self,data,affine,input_path):
        seg = self._convert_to_segmentation(data)
        self._save_img(self.option.get("output_path"),seg,input_path,affine)