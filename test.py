import os
import shutil
import tempfile
from preprocessing.brain_extraction import BrainExtracter
from preprocessing.wrapper import AnimaWrapper


def clean(temp_dir):
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            print(f"Temporary directory '{temp_dir}' has been removed.")
        except Exception as e:
            print(f"Failed to delete temp directory '{temp_dir}': {e}")

def get_image_basename(img_path):
    filename = os.path.basename(img_path)
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    elif filename.endswith(".nii"):
        return filename[:-4]
    else:
        return os.path.splitext(filename)[0]

def find_nii_files(input_path):
        path_list = []
        if os.path.isfile(input_path) and input_path.endswith((".nii.gz",".nii")):
            path_list.append(input_path)
        elif os.path.isdir(input_path):
            for root, _, files in os.walk(input_path):
                for f in files:
                    if f.endswith((".nii.gz",".nii")):
                        path_list.append(os.path.join(root, f))
        return path_list

wrapper = AnimaWrapper()
extracter = BrainExtracter(wrapper)
list = find_nii_files("../ATLAS/test_t1_flair")
output_path = "./"
for img in list:
    print(f"\n Traitement de l'image : {img}")
    temp_dir = tempfile.mkdtemp(prefix="unet_preprocess")
    prefix = os.path.join(temp_dir,get_image_basename(img))
    bet,time = extracter.run(img,prefix)
    shutil.copy(bet,output_path)
    clean(temp_dir)
    print(f"Fini : {img} | Durée bet : {time:.2f} secondes")
