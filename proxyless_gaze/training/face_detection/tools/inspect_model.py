import argparse
import torch
import numpy as np
from thop import profile as thop_profile
from torchprofile import profile_macs

from yolox.exp import get_exp
from yolox.utils import get_model_info


def make_parser():
    parser = argparse.ArgumentParser("YOLOX onnx deploy")
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="experiment description file",
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    return parser


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
    
    # print(prettify(mac_list[:42].sum()+mac_list[136:148].sum()))
    # print(prettify(mac_list[42:90].sum()+mac_list[148:164].sum()))
    # print(prettify(mac_list[90:120].sum()+mac_list[164:180].sum()))
    # print(prettify(mac_list[180:180+21].sum()))
    # print(prettify(mac_list[201:201+21].sum()))
    # print(prettify(mac_list[222:222+21].sum()))
    # print("="*70)
    # print(prettify(param_list[:42].sum()+param_list[136:148].sum()))
    # print(prettify(param_list[42:90].sum()+param_list[148:164].sum()))
    # print(prettify(param_list[90:120].sum()+param_list[164:180].sum()))
    # print(prettify(param_list[180:180+21].sum()))
    # print(prettify(param_list[201:201+21].sum()))
    # print(prettify(param_list[222:222+21].sum()))
    if enable_prettify:
        return prettify(macs), prettify(params)
    else:
        return macs, params

args = make_parser().parse_args()
exp = get_exp(args.exp_file, args.name)
model = exp.get_model()
model.eval()
model.head.decode_in_inference = False
dummy_input = torch.randn(1, 3, 128, 160)

macs, params = profile(dummy_input, model, enable_prettify=True, detail=True)

print("Model:", exp.exp_name)
print("Total MACs:", macs)
print("Total Params:", params)
# print("From YOLOX utils:")
# print(get_model_info(model, exp.test_size))

# model.head.decode_in_inference = True
# outputs = model(dummy_input)
# print(outputs)
