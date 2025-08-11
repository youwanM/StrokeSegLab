import logging
import os
import shutil
import onnxruntime as ort
from utils.config_manager import Config
from utils.path import MODEL_DIR


def update_models() -> None:
    """
    Look for ONNX models in the model directory and update the config.ini file :
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    config = Config()
    models = []
    for _, _, files in os.walk(MODEL_DIR):
        for f in files:
            if f.endswith(".onnx"):
                models.append(f[:-5])
    models_string = ",".join(models)
    if models_string != config.get("default","models"):
        if not models:
            config.set("default", "model", "") 
        elif config.get("default","model") not in models:
            config.set("default","model",models[0]) # If the default model isn't in the models directory, update the config file
        config.set("default","models",models_string) # If the list of models in the models directory is different from the one in the config file, update the config list
        config.clear("ModelChannels") 
        for model in models: # Save the number of input channels for each model in the directory
            model_path = os.path.join(MODEL_DIR, model+".onnx")
            channels = get_input_channels(model_path)
            config.set("ModelChannels",model,str(channels))
        config.save()

def add_model(model_path : str)->str:
    """
    Copy a model file in the models directory
    Raises errors if the model already exist or if the file isn't a .onnx   
    Args:
        model_path (str): Path of the model to import

    Returns:
        str: Basename of the model without the extension
    """
    logger = logging.getLogger()
    config = Config()
    models_str = config.get("default","models")
    models = [m.strip() for m in models_str.split(',')]
    model = os.path.basename(model_path)
    model_name=model[:-5]
    if not model.endswith(".onnx"):
        raise ValueError("Model file must be in .onnx format.")
    if model_name in models:
        raise ValueError(f"Model '{model_name}' already exists in the config.")
    try:
        os.link(model_path, os.path.join(MODEL_DIR, model))
        logger.debug(f"{model} hardlinked successfully")
    except Exception:
        shutil.copy(model_path, os.path.join(MODEL_DIR, model))
        logger.debug(f"{model} copied because hardlink failed")
    return model_name 

def get_input_channels(model_path : str)->int:
    """
    Determine the number of input channels in a model

    Args:
        model_path (str): Path of the model

    Returns:
        int: Number of channels
    """
    session = ort.InferenceSession(model_path)
    input_tensor = session.get_inputs()[0]
    return input_tensor.shape[1]