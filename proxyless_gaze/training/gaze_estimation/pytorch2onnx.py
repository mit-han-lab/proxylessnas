# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import os
import argparse
from models import MyModelv7, MyModelv8
from torch.autograd import Variable
import torch
import onnxsim

parser = argparse.ArgumentParser(description='pytorch2onnx')
parser.add_argument('--torch_model', type=str, required=True)
parser.add_argument('--onnx_model', default="./output/gaze.onnx")
parser.add_argument('--onnx_model_sim',
                    help='Output ONNX model',
                    default="./output/gaze-sim.onnx")
args = parser.parse_args()

print("=====> load pytorch checkpoint...")
model = MyModelv7(arch="proxyless-w0.3-r176_imagenet")
checkpoint = torch.load(args.torch_model, map_location=torch.device('cpu'))
from copy import deepcopy
state_dict = deepcopy(checkpoint['state_dict'])
checkpoint = {}
for key in state_dict:
    if key.startswith("model."):
        checkpoint[key.replace("model.", "")] = state_dict[key]
model.load_state_dict(checkpoint)
model.eval()
print("=====> convert pytorch model to onnx...")
leye = torch.randn((1, 3, 60, 60))
reye = torch.randn((1, 3, 60, 60))
face = torch.randn((1, 3, 120, 120))
dummy_input = (leye, reye, face)
input_names = ["left_eye", "right_eye", "face"]
output_names = ["gaze_pitchyaw"]
torch.onnx.export(model,
                  dummy_input,
                  args.onnx_model,
                  verbose=False,
                  input_names=input_names,
                  output_names=output_names)

print("====> check onnx model...")
import onnx
model = onnx.load(args.onnx_model)
onnx.checker.check_model(model)

print("====> Simplifying...")
model_opt, check = onnxsim.simplify(args.onnx_model)
# print("model_opt", model_opt)
onnx.save(model_opt, args.onnx_model_sim)
print("onnx model simplify Ok!")
