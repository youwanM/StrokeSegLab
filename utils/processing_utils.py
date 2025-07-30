import os
import shutil
from utils.naming import DERIVATIVES, EXTENSIONS, RAWDATA
from utils.option_manager import Option


def get_image_basename(img_path : str)->str:
    """
    Return the basename without the extension of an image path

    Args:
        img_path (str): Image path

    Returns:
        str: Basename without the extension
    """
    name = os.path.basename(img_path)
    for ext in EXTENSIONS:
        if name.endswith(ext):
            name = name[:-len(ext)]
    return name

def move_to_output(img_path : str) ->str:
    """
    Copy an image file to the correct output directory
    Handle BIDS and not BIDS input directory, and also file

    Args:
        img_path (str): Path of the image to copy

    Returns:
        str: Path of the copy
    """
    option = Option()
    subject_name = os.path.basename(img_path).split("_")[0]
    input_path = option.get("input_path")
    if option.get("is_file"):
        if RAWDATA in input_path :
            raw_dir = input_path.split(RAWDATA)[0] # If input is a file in a BIDS directory, it finds the derivatives directory
            output_dir = os.path.join(raw_dir,DERIVATIVES,subject_name,"anat") 
        else:
            output_dir = os.path.dirname(input_path) # If input isn't in a BIDS directory, the copy will be placed in the parent directory of the input file
    else :
        output_dir = os.path.join(input_path,DERIVATIVES,subject_name,"anat")
    os.makedirs(output_dir,exist_ok=True)
    return shutil.copy(img_path,os.path.join(output_dir,os.path.basename(img_path)))

def rm_entity(img_path : str,keyword : str)->str:
    """
    Removes a BIDS entity from the image filename
    The function returns the base filename without the extension and removes the part starting from the entity linked to the keyword (rm_entity('pahtto/sub-0001_acq-T1w.ext') -> sub-0001)
    Args:
        img_path (str): Image path
        keyword (str): Entity keyword to remove

    Returns:
        str: Basename without extension and the entity
    """
    name = get_image_basename(img_path)
    i = name.find(keyword)
    if i ==-1:
        return name
    name = name[:i] # We only keep the string up to the keyword
    if name.endswith("-"): # Handle the case where the entity is _type-keyword, remove after '_'
        name = name.rsplit("_",1)[0]
    name = name.rstrip("_") # Remove the last '_'
    return name