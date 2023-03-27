from turtle import left, right
import torch
import torch.nn as nn
from torch.nn.modules.container import Sequential
from mobilenetv2 import mobilenetv2

pretrained_models = {
    # https://github.com/d-li14/mobilenetv2.pytorch
    'mbv2_1.00':  './pretrained/mobilenetv2_1.0-0c6065bc.pth',
    'mbv2_0.75': './pretrained/mobilenetv2_0.75-dace9791.pth',
    'mbv2_0.50':  './pretrained/mobilenetv2_0.5-eaa6f9ad.pth',
    'mbv2_0.35': './pretrained/mobilenetv2_0.35-b2e15951.pth',
    'mbv2_0.25': './pretrained/mobilenetv2_0.25-b61d2159.pth',
    'mbv2_0.10':  './pretrained/mobilenetv2_0.1-7d1d638a.pth',

    # https://github.com/mit-han-lab/tinyml/tree/master/mcunet
    'mbv2-w0.35-r144_imagenet': './pretrained/mbv2-w0.35-r144_imagenet.pth',
    'mcunet-320kb-1mb_imagenet': './pretrained/mcunet-320kb-1mb_imagenet.pth',
    'mcunet-512kb-2mb_imagenet': './pretrained/mcunet-512kb-2mb_imagenet.pth',
    'proxyless-w0.3-r176_imagenet': './pretrained/proxyless-w0.3-r176_imagenet.pth',
    'proxyless-w0.25-r112_imagenet': './pretrained/proxyless-w0.25-r112_imagenet.pth'
}


def _make_backbone(width_mult=None, last_in_channel=None, last_out_channel=None, arch='mbv2'):
    if arch == 'mbv2':
        ckpt = torch.load(pretrained_models[f"mbv2_{width_mult:.2f}"])
        _mbv2 = mobilenetv2(width_mult=width_mult)
        _mbv2.load_state_dict(ckpt)
        if last_in_channel is None and last_out_channel is None:
            return _mbv2.features
        else:
            _mbv2.features[-1].conv[-2] = nn.Conv2d(
                int(960*width_mult), last_in_channel, 1, 1)
            _mbv2.features[-1].conv[-1] = nn.BatchNorm2d(last_in_channel)
            _mbv2.conv[0] = nn.Conv2d(last_in_channel, last_out_channel, 1, 1)
            _mbv2.conv[1] = nn.BatchNorm2d(last_out_channel)
            return nn.Sequential(*[_mbv2.features, _mbv2.conv])
    else:
        from tinynas.nn.networks import ProxylessNASNets
        import json
        json_file = pretrained_models[arch].replace('.pth', '.json')
        with open(json_file) as f:
            config = json.load(f)
        _model = ProxylessNASNets.build_from_config(config)
        ckpt = torch.load(pretrained_models[arch])
        _model.load_state_dict(ckpt['state_dict'], strict=False)
        return _model, config['classifier']['in_features']


class MyModelv7(nn.Module):
    def __init__(self, arch, **kwargs):
        super(MyModelv7, self).__init__()
        self.eye_channel, eye_out_features = _make_backbone(arch=arch)
        self.attention_branch, attention_out_features = _make_backbone(
            arch=arch)
        _face_backbone, face_out_features = _make_backbone(arch=arch)
        self.face_channel = Sequential(
            _face_backbone,
            nn.Dropout(0.2),
            nn.Linear(face_out_features, 256),
            nn.ReLU(inplace=True))  # add relu
        self.leye_fc = Sequential(
            nn.Dropout(0.2),
            nn.Linear(eye_out_features, 256),
            nn.ReLU(inplace=True))
        self.reye_fc = Sequential(
            nn.Dropout(0.2),
            nn.Linear(eye_out_features, 256),
            nn.ReLU(inplace=True))
        self.leye_attention_fc = Sequential(
            nn.Linear(attention_out_features, 256),
            nn.Sigmoid())
        self.reye_attention_fc = Sequential(
            nn.Linear(attention_out_features, 256),
            nn.Sigmoid())
        self.regressor = nn.Sequential(
            nn.Linear(256*3, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2))

    def forward(self, leye, reye, face):
        left_eye_feature = self.eye_channel(leye)
        right_eye_feature = self.eye_channel(reye)
        left_eye_feature = self.leye_fc(left_eye_feature)
        right_eye_feature = self.reye_fc(right_eye_feature)

        left_eye_attention = self.attention_branch(leye)
        right_eye_attention = self.attention_branch(reye)
        left_eye_attention = self.leye_attention_fc(left_eye_attention)
        right_eye_attention = self.reye_attention_fc(right_eye_attention)

        left_eye_feature = left_eye_feature * left_eye_attention
        right_eye_feature = right_eye_feature * right_eye_attention
        face_feature = self.face_channel(face)
        x = torch.cat([face_feature, left_eye_feature, right_eye_feature], -1)
        x = self.regressor(x)
        return x


def pretty_print(x):
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


if __name__ == "__main__":
    from torchprofile import profile_macs
    from thop import profile
    # (60, 60) (120, 120)
    leye = torch.randn((1, 3, 60, 60)).cuda()
    reye = torch.randn((1, 3, 60, 60)).cuda()
    face = torch.randn((1, 3, 120, 120)).cuda()
    input = (leye, reye, face)
    model = MyModelv7(arch="proxyless-w0.25-r112_imagenet").cuda()
    macs = profile_macs(model, args=input)
    _, params = profile(model, inputs=input, verbose=False)
    print(pretty_print(macs), pretty_print(params))
    print(model(*input))
