import os
import onnx
from onnxconverter_common import float16

from utils.path import MODEL_DIR

input_path = os.path.join(MODEL_DIR,"model-T1FLAIR.onnx")
output_path = os.path.join(MODEL_DIR,"model_fp16.onnx")

model = onnx.load(input_path)
model_fp16 = float16.convert_float_to_float16(model)
onnx.save(model_fp16, output_path)