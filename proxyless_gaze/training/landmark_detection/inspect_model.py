import argparse
import torch
import numpy as np
from thop import profile as thop_profile
from torchprofile import profile_macs
from models.pfld import PFLDInference, PFLDInferenceOriginal

def prettify(x):
    if x is None:
        return ""
    if x < 1000:
        return str(round(x, 3))
    elif x < 1000000:
        return str(round(x/1000, 3))+"K"
    elif x < 1000000000:
        return str(round(x/1000000, 3))+"M"
    else:
        return str(round(x/1000000000, 3))+"G"

def profile(input, model, enable_prettify=False, detail=False):
    macs = profile_macs(model, input)
    _, params = thop_profile(model, inputs=(input,), verbose=False)
    if detail:
        info = profile_macs(model, input, reduction=None)
        mac_list = []
        param_list = []
        print(f'{"Index":<10} {"Operation":<30} {"Output Shape":<20} {"MACs":<10} {"Params"}')
        for i, op in enumerate(info):
            print(f'{i:<10} {op.operator:<30} {str(op.outputs[0].shape):<20} {prettify(info[op]):<10} {prettify(np.prod(op.inputs[1].shape))}')
            mac_list.append(info[op])
            param_list.append(np.prod(op.inputs[1].shape))
    mac_list = np.array(mac_list)
    param_list = np.array(param_list)
    
    # print(prettify(mac_list[:42].sum()))
    # print(prettify(mac_list[42:90].sum()))
    # print(prettify(mac_list[90:122].sum()))
    # print("="*70)
    # print(prettify(param_list[:42].sum()))
    # print(prettify(param_list[42:90].sum()))
    # print(prettify(param_list[90:122].sum()))

    print(prettify(mac_list[:82].sum()))
    print("="*70)
    print(prettify(param_list[:82].sum()))
    if enable_prettify:
        return prettify(macs), prettify(params)
    else:
        return macs, params

model = PFLDInferenceOriginal()
model.eval()
dummy_input = torch.randn(1, 3, 112, 112)
macs, params = profile(dummy_input, model, enable_prettify=True, detail=True)

print("Total MACs:", macs)
print("Total Params:", params)
