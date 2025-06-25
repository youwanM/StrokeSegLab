from preprocessing.wrapper import AnimaWrapper
import os
import tempfile
import shutil
import time

class BrainExtraction:

    def __init__(self,atlas_path="./anima_scripts/atlas.nrrd",atlas_mask_path="./anima_scripts/atlas_brain_mask.nrrd"):
        self.wrapper = AnimaWrapper()
        self.atlas = atlas_path
        self.atlas_mask = atlas_mask_path
        self.temp_dir = tempfile.mkdtemp(prefix="anima_brain_extract_")
        self.pyramid_option = ["-p", "4", "-l", "1"]
        

    def run(self,img_path):
        start = time.time()
        prefix = os.path.join(self.temp_dir,self._get_image_basename(img_path))
        input_dir = os.path.dirname(img_path)
        output_prefix = os.path.join(input_dir,self._get_image_basename(img_path))
        brainMask = output_prefix + "_brainMask.nii.gz"
        maskedBrain = output_prefix + "_masked.nii.gz"

        command = ["animaPyramidalBMRegistration","-m",self.atlas,"-r",img_path,"-o",prefix+"_rig.nrrd","-O",prefix+"_rig_tr.txt","--sp","3"] + self.pyramid_option
        self.wrapper.run(command)

        command = ["animaPyramidalBMRegistration", "-m", self.atlas, "-r", img_path, "-o", prefix + "_aff.nrrd", "-O", prefix + "_aff_tr.txt", "-i", prefix + "_rig_tr.txt", "--sp", "3", "--ot","2"] + self.pyramid_option
        self.wrapper.run(command)

        command = ["animaCreateImage", "-g", self.atlas, "-b", "1", "-o", prefix + "_baseCropMask.nrrd"]
        self.wrapper.run(command)

        command = ["animaTransformSerieXmlGenerator", "-i", prefix + "_aff_tr.txt","-o", prefix + "_aff_tr.xml"]
        self.wrapper.run(command)

        command = ["animaApplyTransformSerie", "-i", prefix + "_baseCropMask.nrrd","-t", prefix + "_aff_tr.xml", "-g", img_path, "-o",prefix + "_cropMask.nrrd", "-n", "nearest"]
        self.wrapper.run(command)

        command = ["animaMaskImage", "-i", img_path, "-m", prefix + "_cropMask.nrrd","-o", prefix + "_c.nrrd"]
        self.wrapper.run(command)

        command = ["animaDenseSVFBMRegistration", "-r", prefix + "_c.nrrd", "-m", prefix + "_aff.nrrd","-o", prefix + "_nl.nrrd", "-O", prefix + "_nl_tr.nrrd", "--tub", "2"] + self.pyramid_option
        self.wrapper.run(command)

        command = ["animaTransformSerieXmlGenerator", "-i", prefix + "_aff_tr.txt", "-i",prefix + "_nl_tr.nrrd", "-o", prefix + "_nl_tr.xml"]
        self.wrapper.run(command)

        command = ["animaApplyTransformSerie", "-i", self.atlas_mask, "-t", prefix + "_nl_tr.xml", "-g", img_path, "-o",prefix + "_rough_brainMask.nrrd", "-n", "nearest"]
        self.wrapper.run(command)

        command = ["animaMaskImage", "-i", img_path, "-m", prefix + "_rough_brainMask.nrrd", "-o",prefix + "_rough_masked.nrrd"]
        self.wrapper.run(command)

        brainImageRoughMasked = prefix + "_rough_masked.nrrd"

        command = ["animaConvertImage", "-i", brainImageRoughMasked, "-o", prefix + "_masked.nrrd"]
        self.wrapper.run(command)
        command = ["animaConvertImage", "-i", prefix + "_rough_brainMask.nrrd", "-o", prefix + "_brainMask.nrrd"]
        self.wrapper.run(command)

        command = ["animaConvertImage", "-i", prefix + "_masked.nrrd", "-o", maskedBrain]
        self.wrapper.run(command)
        command = ["animaConvertImage", "-i", prefix + "_brainMask.nrrd", "-o", brainMask]
        self.wrapper.run(command)
        end = time.time()
        elapsed = end - start
        print(f"Temps passé au skull strip : {elapsed:.2f} secondes")

    
    def clean(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def _get_image_basename(self,img_path):
        filename = os.path.basename(img_path)
        if filename.endswith(".nii.gz"):
            return filename[:-7]  # remove ".nii.gz"
        elif filename.endswith(".nii"):
            return filename[:-4]  # remove ".nii"
        else:
            return os.path.splitext(filename)[0]