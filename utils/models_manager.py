import os
import shutil
import onnxruntime as ort
from utils.config_manager import Config
from utils.path import MODEL_DIR


def update_models():
    config = Config()
    models = []
    for _, _, files in os.walk(MODEL_DIR):
        for f in files:
            if f.endswith(".onnx"):
                models.append(f[:-5])
    models_string = ",".join(models)
    if models_string != config.get("default","models"):
        if config.get("default","model") not in models:
            config.set("default","model",models[0])
        config.set("default","models",models_string)
        config.clear("ModelChannels")
        for model in models:
            model_path = os.path.join(MODEL_DIR, model+".onnx")
            session = ort.InferenceSession(model_path)
            input_tensor = session.get_inputs()[0]
            channels = input_tensor.shape[1]
            config.set("ModelChannels",model,str(channels))
        config.save()

def add_model(model_path):
    config = Config()
    models_str = config.get("default","models")
    models = [m.strip() for m in models_str.split(',')]
    model = os.path.basename(model_path)
    model_name=model[:-5]
    if not model.endswith(".onnx"):
        raise ValueError("Model file must be in .onnx format.")
    if model_name in models:
        raise ValueError(f"Model '{model_name}' already exists in the config.")
    shutil.copy(model_path,os.path.join(MODEL_DIR,model))
    return model_name 