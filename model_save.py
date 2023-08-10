import torch
import torchvision
import onnx
import onnx
import onnx_tf
import tensorflow as tf

model = torch.load('trained_model.pt')

torch.onnx.export(model,               
                  torch.randn(1, 3, 224, 224), # dummy input (required)
                  "resnet18.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True) # store the trained parameter weights inside the model file

model = onnx.load("train_onnx_model")
onnx.checker.check_model(model)

tf_model = onnx_tf.convert_from_onnx(model)

tf.io.write_graph(tf_model, ".", "train_onnx_model", as_text=False)
