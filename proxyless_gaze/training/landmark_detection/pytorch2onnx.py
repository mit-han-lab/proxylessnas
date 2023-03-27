# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import os
import argparse
from models.pfld import PFLDInference
from torch.autograd import Variable
import torch
import onnxsim

parser = argparse.ArgumentParser(description='pytorch2onnx')
parser.add_argument('--torch_model', type=str, required=True)
parser.add_argument('--onnx_model', default="./output/pfld.onnx")
parser.add_argument('--onnx_model_sim',
                    help='Output ONNX model',
                    default="./output/pfld-sim.onnx")
args = parser.parse_args()

print("=====> load pytorch checkpoint...")
checkpoint = torch.load(args.torch_model, map_location=torch.device('cpu'))
if "state_dict" in checkpoint:
    pfld_backbone = PFLDInference(98)
    from copy import deepcopy
    state_dict = deepcopy(checkpoint['state_dict'])
    checkpoint = {}
    for key in state_dict:
        if key.startswith("pfld_backbone."):
            checkpoint[key.replace("pfld_backbone.", "")] = state_dict[key]
    pfld_backbone.load_state_dict(checkpoint)
else:
    pfld_backbone = PFLDInference()
    pfld_backbone.load_state_dict(checkpoint['pfld_backbone'])
print("PFLD bachbone:", pfld_backbone)
pfld_backbone.eval()

print("=====> convert pytorch model to onnx...")
dummy_input = Variable(torch.randn(1, 3, 112, 112))
input_names = ["face_image"]
output_names = ["headpose_feature", "facial_landmark"]

_, torch_out = pfld_backbone(dummy_input)

torch.onnx.export(pfld_backbone,
                  dummy_input,
                  args.onnx_model,
                  verbose=True,
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

import onnxruntime
import numpy as np
ort_session = onnxruntime.InferenceSession("./output/pfld-sim.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
_, ort_outs = ort_session.run(None, ort_inputs)
print(ort_outs.shape)
print(torch_out.shape)
print(ort_outs.reshape(-1)[:20])
print(torch_out.reshape(-1)[:20])
# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out), ort_outs, rtol=1e-03, atol=1e-05)
