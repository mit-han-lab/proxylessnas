# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware [[arXiv]](https://arxiv.org/abs/1812.00332) [[Poster]](https://hanlab.mit.edu/files/proxylessNAS/figures/ProxylessNAS_iclr_poster_final.pdf)
```bash
@inproceedings{
  cai2018proxylessnas,
  title={Proxyless{NAS}: Direct Neural Architecture Search on Target Task and Hardware},
  author={Han Cai and Ligeng Zhu and Song Han},
  booktitle={International Conference on Learning Representations},
  year={2019},
  url={https://arxiv.org/pdf/1812.00332.pdf},
}
```

## News
- Next generation of ProxylessNAS: [Once-for-All](https://github.com/mit-han-lab/once-for-all) (First place in the 3rd and 4th [Low-Power Computer Vision Challenge](https://lpcv.ai/competitions/2019)). 
- First place in the Visual Wake Words Challenge, TF-lite track, @CVPR 2019
- Third place in the Low Power Image Recognition Challenge (LPIRC), classification track, @CVPR 2019

## Performance

Without any proxy, directly and efficiently search neural network architectures on your target **task** and **hardware**! 

![](https://hanlab.mit.edu/files/proxylessNAS/figures/proxyless_nas.png)

<p align="center">
    <img src="https://hanlab.mit.edu/files/proxylessNAS/figures/proxyless_bar.png" width="80%" />
</p>

![](https://hanlab.mit.edu/files/proxylessNAS/figures/proxyless_compare.png)

<table>
<tr>
    <th> Mobile settings </th><th> GPU settings </th>
</tr>
<tr>
    <td>
    <img src="https://hanlab.mit.edu/files/proxylessNAS/figures/proxyless_vs_mobilenet.png" width="100%" />
    </td>
<td>

| Model                | Top-1    | Top-5    | Latency | 
|----------------------|----------|----------|---------| 
| MobilenetV2          | 72.0     | 91.0     | 6.1ms   |
| ShufflenetV2(1.5)    | 72.6     | -        | 7.3ms   |
| ResNet-34            | 73.3     | 91.4     | 8.0ms   |
| MNasNet(our impl)    | 74.0     | 91.8     | 6.1ms   | 
| ProxylessNAS (GPU)   | 75.1     | 92.5     | 5.1ms   |

</td>
</tr>
<tr>
    <th> ProxylessNAS(Mobile) consistently outperforms MobileNetV2 under various latency settings.  </th>
    <th> ProxylessNAS(GPU) is 3.1% better than MobilenetV2 with 20% faster. </th>
</tr> 

</td></tr> </table>


## Specialization

People used to deploy one model to all platforms, but this is not good. To fully exploit the efficiency, we should specialize architectures for each platform.

![](https://hanlab.mit.edu/files/proxylessNAS/figures/specialization.jpg)
[![Watch the video](https://hanlab.mit.edu/files/proxylessNAS/figures/specialized_archs.png)](https://hanlab.mit.edu/files/proxylessNAS/visualization.mp4)

## Requirements
- Pytorch 1.0
- Python 3.6+

```bash
# Train a model: <path>/run.config; <path>/net.config
python python imagenet_run_exp.py --path <path> --train

# Train Proxyless (GPU)
python imagenet_run_exp.py --path Exp/proxyless_gpu --train --net proxyless_gpu --dropout 0.3

# Train Proxyless (CPU)
python imagenet_run_exp.py --path Exp/proxyless_cpu --train --net proxyless_cpu --dropout 0.2

# Train Proxyless (Mobile)
python imagenet_run_exp.py --path Exp/proxyless_mobile --train --net proxyless_mobile --dropout 0.1

# Train Proxyless (Mobile-14)
python imagenet_run_exp.py --path Exp/proxyless_mobile_14 --train --net proxyless_mobile_14 --dropout 0.3
```

```bash
# Eval Proxyless (GPU)
python imagenet_run_exp.py --path Exp/proxyless_gpu --net proxyless_gpu

# Eval Proxyless (CPU)
python imagenet_run_exp.py --path Exp/proxyless_cpu --net proxyless_cpu

# Eval Proxyless (Mobile)
python imagenet_run_exp.py --path Exp/proxyless_mobile --net proxyless_mobile

# Eval Proxyless (Mobile-14)
python imagenet_run_exp.py --path Exp/proxyless_mobile_14 --net proxyless_mobile_14
```

```bash
# Architecture Search
please refer to imagenet_arch_search.py
```