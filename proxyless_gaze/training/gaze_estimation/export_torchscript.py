import torch
from models import MyModelv7
import argparse

parser = argparse.ArgumentParser(description='export model to torchscript')
parser.add_argument('--torch_model', type=str, required=True)
parser.add_argument("--output_name", type=str, default="./output/gaze.pt")
args = parser.parse_args()


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

leye = torch.randn((1, 3, 60, 60))
reye = torch.randn((1, 3, 60, 60))
face = torch.randn((1, 3, 120, 120))
dummy_input = (leye, reye, face)

mod = torch.jit.trace(model, dummy_input)
mod.save(args.output_name)