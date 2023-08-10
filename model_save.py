import torch
import torchvision
import onnx
import onnx
import onnx_tf
import tensorflow as tf

model = torch.load('trained_model.pt')

torch.onnx.export(model,               
                  torch.randn(1, 3, 224, 224), 
                  "trained_onnx_model",
                  export_params=True)

model = onnx.load("trained_onnx_model")
onnx.checker.check_model(model)

tf_model = onnx_tf.convert_from_onnx(model)
tf.io.write_graph(tf_model, ".", "trained_onnx_model", as_text=False)

model_json = tf_model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

with open("model.json", "r") as json_file:
    loaded_model_json = json_file.read()

loaded_model = tf.keras.models.model_from_json(loaded_model_json)
