dependencies = ['torch', 'torchvision']

from torch.hub import load_state_dict_from_url
from proxyless_nas.model_zoo import proxyless_cpu, proxyless_gpu, proxyless_mobile, proxyless_mobile_14
