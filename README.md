## Hierarchical Pruning of Lightweight Multiscale Edge-Aware Transformer for Real-Time Insulator Defect Detection Model

## Setup

### Getting Started
Installation (to install pytorch cf. https://pytorch.org/get-started/locally/):
```shell
conda create -n npsr python=3.11
conda activate npsr
pip install torch torchvision torchaudio
```

## Training


usage:
```shell
python trian.py
```

## Visualization

usage:
```shell
python val.py
```
## Datasets

### INSULATOR FAULTSDETECTION  dataset
You can get the INSULATOR FAULTSDETECTION dataset by filling out the form at:
https://universe.roboflow.com/project-vmgqx/insulator-faults-detection

### VOC dataset
vim trian.py:
```shell
import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR


if __name__ == '__main__':
    model = RTDETR('.../rtdetr_pytorch/configs/rtdetr/HLME-DETR.yaml')
    # model.load('') # loading pretrain weights
    model.train(data='.../data.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=32,
                workers=4, 
                # device='0', 
                # resume='', # last.pt path
                project='runs/train',
                name='exp',
                )
```

### SFID dataset
Dataset downloadable at:
https://ihepbox.ihep.ac.cn/ihepbox/index.php/s/adTHe1UPu0Vc7vI/download

