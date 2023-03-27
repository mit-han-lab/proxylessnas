# ProxylessGaze: Real-time Gaze Estimation with ProxylessNAS

This folder contains code to train and deploy face detection, facial landmark detection and gaze estimation models on various platforms including **Intel CPU, ARM CPU, Qualcomm GPU** using various deployment tools including **ONNX, TVM and SNPE SDK**. The whole pipeline can run at a real time on Intel CPU (~80FPS on i5 8300H), Raspberry Pi 4 (~30FPS) and Qualcomm mobile GPU (~50FPS on 8Gen1) with satisfactory accuracy.

<pre class="vditor-reset" placeholder="" contenteditable="true" spellcheck="false"><p data-block="0"><img src="https://s2.loli.net/2022/10/19/FmlsK1v2GATB5LC.jpg" alt="image.jpg"/></p></pre>

## Quick Demo

Adreno 619 GPU with SNPE SDK:

<div align=center>
<img src="assets/android_demo.gif" style="width:100%"></img>
</div>

Raspberry Pi 4 with TVM:

<div align=center>
<img src="assets/rpi4_demo.gif" style="width:100%"></img>
</div>

## Training

### Face Detection

Our face detection model training is based on [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX). The training code is located at `training/face_detection`.

#### Installation

```bash
pip3 install -v -e .
cd widerface
python3 setup.py build_ext --inplace
```

#### Prepare training data

Download widerface dataset from [here](http://shuoyang1213.me/WIDERFACE/).

Convert widerface to coco format:

```bash
python3 tools/convert.py -i INPUT_DATASET_PATH -o OUTPUT_DATASET_PATH
```

#### Train and export

Train the model using 2 GPUs with batchsize of 64 (you can set the GPU numbers and batchsize on your own):

```bash
python3 -m yolox.tools.train -f exps/widerface/proxyless_160x128_v2.py -d 2 -b 64
```

After training, export the model to onnx format:

```bash
python3 tools/export_onnx.py --output-name yolox.onnx -f exps/widerface/proxyless_160x128_v2.py -c CHECKPOINT_PATH
```

Or, export the model to torchscript format:

```bash
python3 tools/export_torchscript.py --output-name yolox.pt -f exps/widerface/proxyless_160x128_v2.py -c CHECKPOINT_PATH
```

### Facial Landmark Detection

Our facial landmark detection model training is based on [PFLD](https://github.com/polarisZhao/PFLD-pytorch). The training code is located at `training/landmark_detection`.

#### Installation

```bash
pip3 install -r requirements.txt
```

#### Prepare training data

We train PFLD using [WFLW](https://wywu.github.io/projects/LAB/WFLW.html) and [facescrub](http://vintage.winklerbros.net/facescrub.html) datasets.

Download them and run the following script to convert them to correct format:

```bash
cd data
python3 SetPreparationFacescrub.py
python3 SetPreparationWFLW.py
```

#### Train and export

```bash
python3 train.py
```

Use --help flag to see the training options.

After training, export the model to onnx format:

```bash
python3 pytorch2onnx.py --torch_model CHECKPOINT_PATH
```

### Gaze Estimation

The training code for gaze estimation is located at `training/gaze_estimation`.

#### Installation

```bash
pip3 install torch torchvision torchaudio pytorch-lightning
```

#### Prepare training data

[ETH-XGaze](https://ait.ethz.ch/projects/2020/ETH-XGaze/) dataset is used for training. We use 224x224 images from the dataset.

Download the images and convert the data to correct format:

```bash
cd utils
python3 preprocess_xgaze.py
```

#### Train and export

```bash
python3 train.py
```

After training, export the model to onnx format:

```bash
python3 pytorch2onnx.py --torch_model CHECKPOINT_PATH
```

## Deployment

We provide deployment code and demo for:

1. ONNX runtime on Intel CPU.
2. TVM on Raspberry Pi 4.
3. Android app on Qualcomm GPU using [SNPE SDK](https://developer.qualcomm.com/sites/default/files/docs/snpe/overview.html).

### ONNX Runtime

See code located at `deployment/onnx`.

Install requirements:

```bash
pip3 install -r requirements.txt
```

Run the demo:

```
python3 main.py
```

### TVM

You need to first install tvm following the [official guidance](https://tvm.apache.org/docs/install/index.html).

#### Build TVM engines

You can build the tvm engines for the three models on your own in this section, or you can safely skip this and use the engines we built.

Go to `build_engine/${task_name}` folder, and run:

```bash
# tune for the optimal configuration on RPI4
python3 convert.py
# build the engine using the log.json obtained above
python3 build_engine.py
```

We use a RPI4 farm to speed up tuning. For more information about how to setup this, refer to this [tutorial](https://tvm.apache.org/docs/how_to/tune_with_autoscheduler/tune_network_arm.html#start-rpc-tracker).

#### Run demo

```
cd demo
python3 demo.py
```

### Android App with SNPE SDK

See [README.md](deployment/android/README.md) located at `deployment/android`.

## References

https://github.com/Megvii-BaseDetection/YOLOX

https://github.com/polarisZhao/PFLD-pytorch

https://github.com/mit-han-lab/tinyengine

## Contributor
[Junyan Li](https://github.com/senfu)

## Citation
If you find this work useful for you, please consider citing our paper
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
