import os
import shutil
from utils.naming import DERIVATIVES, EXTENSIONS, RAWDATA
from utils.option_manager import Option


def get_image_basename(img_path):
    name = os.path.basename(img_path)
    for ext in EXTENSIONS:
        if name.endswith(ext):
            name = name[:-len(ext)]
    return name

def move_to_output(img_path):
    option = Option()
    subject_name = os.path.basename(img_path).split("_")[0]
    input_path = option.get("input_path")
    if option.get("is_file"):
        if RAWDATA in input_path :
            raw_dir = input_path.split(RAWDATA)[0]
            output_dir = os.path.join(raw_dir,DERIVATIVES,subject_name,"anat")
        else:
            output_dir = os.path.dirname(input_path)
    else :
        output_dir = os.path.join(input_path,DERIVATIVES,subject_name,"anat")
    os.makedirs(output_dir,exist_ok=True)
    return shutil.copy(img_path,os.path.join(output_dir,os.path.basename(img_path)))

def rm_entity(img_path,keyword):
    name = get_image_basename(img_path)
    i = name.find(keyword)
    if i ==-1:
        return name
    name = name[:i]
    if name.endswith("-"):
        name = name.rsplit("_",1)[0]
    name = name.rstrip("_")
    return name