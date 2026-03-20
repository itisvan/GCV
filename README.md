# PhantomCV (YOLOv12)
PhantomCV is an open-source Computer Vision script for use with Titan Two and Gtuner. Phantom makes use of a **YOLOv12** model to detect objects on screen and then calculates a trajectory for the right stick that it sends to a separate Gtuner script for interpretation. Based on [BradyMeighan/PhantomCV](https://github.com/BradyMeighan/PhantomCV), upgraded to YOLOv12.

## Preview
[![Preview](https://img.youtube.com/vi/lcsM_D6omaw/0.jpg)](https://www.youtube.com/watch?v=lcsM_D6omaw)

## Installation Requirements
 - [Python 3.10+](https://www.python.org/downloads/) (Python 3.10 or newer recommended for YOLOv12 / Ultralytics)
 - [pip](https://bootstrap.pypa.io/get-pip.py)
 - [Visual Studio Community](https://visualstudio.microsoft.com/downloads/)
 - [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
 - [Gtuner](https://www.consoletuner.com/titan-two-downloads/)

### Installation
Assuming you have all the requirements listed above downloaded and installed, run the `4.install-cv.bat` script on your machine to install all the required Python modules.

> **Note:** This version uses the [Ultralytics](https://github.com/ultralytics/ultralytics) unified API which supports YOLOv12. The old `yolov5` PyPI package is no longer required.

> **Important — delete any old compiled module:** If you previously ran the original BradyMeighan/PhantomCV and compiled `phantomcv_helper.py` with Nuitka, a `phantomcv_helper.pyd` file may exist in your working directory. Python loads `.pyd` files **before** `.py` files, so the old compiled code (which used `torch.hub.load` / YOLOv5) would be executed instead of the new Ultralytics-based code, causing the error `'Detect' object has no attribute 'grid'`. **Delete `phantomcv_helper.pyd` (and any `phantomcv_helper.cp*.pyd` variants) before running.**

## How To Make Your Own Changes

### Compile phantomcv_helper.py with Nuitka
Nuitka is a Python package used to compile `phantomcv_helper.py` as a `.pyd` file. The conversion to C is more efficient and provides a performance boost. To compile with Nuitka, run `helpermodule_maker.bat` and have `phantomcv_helper.py` in the same directory.

### YOLOv12 Model
To use a YOLOv12 model:
1. Train or download a YOLOv12 `.pt` weights file.
2. Place the `.pt` file in the same directory as the scripts.
3. Set `modelName` in `settings.py` to the filename (without `.pt` extension).

The Ultralytics API supports loading YOLOv12 weights directly:
```python
from ultralytics import YOLO
model = YOLO('your_model.pt')
```

### Dataset
Here is a link to the dataset used to train the original `warzone.pt` model. It was trained on over 9,000 images for 6,000 generations at an image size of 416.

[Google Drive Link](https://drive.google.com/file/d/1F2vXIlsopzv8AQtsGaopYcSvoexcIcS6/view?usp=sharing)

## Disclaimer

Please do not use this tool to gain an advantage in online gameplay. However, feel free to enjoy its features in offline gameplay, and share your thoughts if you're interested in making the tool better for those who suffer from disabilities. This project is being released as a learning tool to help new programmers learn about object detection and OpenCV. Thank you!

## Donate
Bitcoin - 1Fwdk5fkhZ1Y3uyb2HKywHyHQuddiik6Xu

Ethereum - 0x794067Aa418A6bBe774eE313140cCDFd35C2a148

Dogecoin - DS8q14xNtRD3FZAzpmTrGfJ88Ain61wXor

Ripple - rMVLwS4A5onNvcCPYAeqS2fq3DQg6DRr7L

Cardano - addr1q8u9add5zjczkulmexmm4v3hwjqqra522544c046096l7qahx5a2eqnz26pu84yfs39j4rspz3xaqyjs2el0ldx3fxqqqsnsu0
