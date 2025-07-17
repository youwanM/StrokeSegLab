import time
from manager.path import ATLAS_DIR
import os
class BrainExtracter:

    def __init__(self,wrapper):
        self.wrapper = wrapper
        self.atlas = os.path.join(ATLAS_DIR,"atlas.nrrd")
        self.atlas_mask = os.path.join(ATLAS_DIR,"atlas_brain_mask.nrrd")
        self.pyramid_option = ["-p", "4", "-l", "1"]
        

    def run(self,img_path,prefix):
        start = time.time()
        brainMask = prefix + "_brainMask.nii.gz"
        maskedBrain = prefix + "_BET.nii.gz"

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

        return maskedBrain, elapsed
    
