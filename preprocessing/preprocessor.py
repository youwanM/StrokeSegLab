import nibabel
from manager.option_manager import Option
from preprocessing.normalization import ZScoreNormalization
from preprocessing.resampling import Resampler
import numpy as np

class Preprocessor:
    def __init__(self):
        self.option = Option()
        self.resampler = Resampler()
    
    def _load_img(self,img_path):
        img = nibabel.load(img_path)
        affine = img.affine
        spacing = img.header.get_zooms()
        data = img.get_fdata().astype("float32")
        if data.ndim==3:
            data = np.expand_dims(data,axis=0)
        return data, spacing, affine
    
    def run(self,img_path):
        data, spacing, affine = self._load_img(img_path)
        new_spacing = (1.0, 1.0, 1.0)
        data = self.resampler.run(data,spacing,new_spacing)
        data = ZScoreNormalization.run(data)
        return data, affine
