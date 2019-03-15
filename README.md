- March 15, 2019: for our most updated work on model compression and acceleration, please reference: 

	[ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware](https://arxiv.org/pdf/1812.00332.pdf) (ICLR’19)

	[AMC: AutoML for Model Compression and Acceleration on Mobile Devices](https://arxiv.org/pdf/1802.03494.pdf) (ECCV’18)

	[HAQ: Hardware-Aware Automated Quantization](https://arxiv.org/pdf/1811.08886.pdf)  (CVPR’19, oral)
	
	[Defenstive Quantization: When Efficiency Meets Robustness](https://openreview.net/pdf?id=ryetZ20ctX) (ICLR'19)
  
# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
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

Without any proxy, directly search neural network architectures on your target **task** and **hardware**! 

![](https://hanlab.mit.edu/files/proxylessNAS/figures/proxyless_nas.png)
[Website](https://hanlab.mit.edu/projects/proxylessNAS/), [arXiv](https://arxiv.org/abs/1812.00332)

## Requirements
* PyTorch 0.3.1 or Tensorflow 1.5
* Python 3.6+

## Updates
* Dec-21-2018: TensorFlow pretrained models are released.
* Dec-01-2018: PyTorch pretrained models are released. 

## Performance
<table>
<tr><th> Mobile settings </th><th> GPU settings </th></tr>
<tr><td> 

| Model                | Top-1    | Top-5    | Latency | FLOPs | 
|----------------------|----------|----------|---------|--------|
| MobilenetV1          | 70.6     | 89.5     | 113ms   | 575M   |
| MobilenetV2          | 72.0     | 91.0     | 75ms    | 300M   |
| MNasNet(our impl)    | 74.0     | 91.8     | 79ms    | 317M   |
| ProxylessNAS (mobile)| 74.6     | 92.2     | 78ms    | 320M   |
| ProxylessNAS (mobile_14) | 76.7 | 93.3     | 147ms   | 581M   |

</td><td>

| Model                | Top-1    | Top-5    | Latency | 
|----------------------|----------|----------|---------| 
| MobilenetV2          | 72.0     | 91.0     | 6.1ms   |
| ShufflenetV2(1.5)    | 72.6     | -        | 7.3ms   |
| ResNet-34            | 73.3     | 91.4     | 8.0ms   |
| MNasNet(our impl)    | 74.0     | 91.8     | 6.1ms   | 
| ProxylessNAS (GPU)   | 75.1     | 92.5     | 5.1ms   |

</td><td>
<tr>
    <th> 2.6% better than MobilenetV2 with same speed. </th>
    <th> 3.1% better than MobilenetV2 with 20% faster. </th>
</tr>

</td></tr> </table>

<p align="center">
    <img src="https://hanlab.mit.edu/files/proxylessNAS/figures/proxyless_vs_mobilenet.png" width="50%" />
    </br>
    <a> ProxylessNAS consistently outperforms MobileNetV2 under various latency settings. </a>
</p>

## Specialization

People used to deploy one model to all platforms, but this is not good. To fully exploit the efficiency, we should specialize architectures for each platform.

![](https://hanlab.mit.edu/files/proxylessNAS/figures/specialization.jpg)
![](https://hanlab.mit.edu/files/proxylessNAS/figures/specialized_archs.png)

Please refer to our [paper](https://arxiv.org/abs/1812.00332) for more results.
 
# How to use / evaluate 
* Use
    ```python
    # pytorch 
    from proxyless_nas import proxyless_cpu, proxyless_gpu, proxyless_mobile, proxyless_mobile_14
    net = proxyless_cpu(pretrained=True) # Yes, we provide pre-trained models!
    ```
    ```python
    # tensorflow
    from proxyless_nas_tensorflow import proxyless_cpu, proxyless_gpu, proxyless_mobile, proxyless_mobile_14
    tf_net = proxyless_cpu(pretrained=True)
    ```
* Evaluate

    `python eval.py --path 'Your path to imagent' --arch proxyless_cpu  # pytorch`
    
    `python eval_tf.py --path 'Your path to imagent' --arch proxyless_cpu  # tensorflow`
