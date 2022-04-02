## Object recognition with GCNs

[doc](https://docs.google.com/document/d/1yKBVNr90n2kipyQP4itzt3zvdUfGeSTm2qZ-MJNa8sg/edit?usp=sharing)


## 1. Specification of dependencies

This code requires Python 3.8.12. Install the following packages:

- numpy
- torchnet
- torch-1.8.2
- torchvision-0.9.2
- tqdm

## 2a. Training code

Training and evaluation options:
- `lr`: learning rate
- `lrp`: factor for learning rate of pretrained layers. The learning rate of the pretrained layers is `lr * lrp`
- `batch-size`: number of images per batch
- `image-size`: size of the image
- `epochs`: number of training epochs
- `evaluate`: evaluate model on validation set
- `resume`: path to checkpoint

For trainval on Pascal VOC 2007, run:
```sh
python3 demo_voc2007_gcn.py data/voc --image-size 448 --batch-size 4
``` 

For trainval on COCO 2014, run:
```sh
python3 demo_coco_gcn.py data/coco --image-size 448 --batch-size 8
```

## 2b Evaluation code
For testing on Pascal VOC 2007, run:
```sh
python3 demo_voc2007_gcn.py data/voc --image-size 448 --resume checkpoint/voc2007/model_best_91.8316.pth.tar --evaluate
```

For testing on COCO 2014, run:
```sh
python3 demo_coco_gcn.py data/coco --image-size 448 --resume checkpoint/coco/model_best_80.2723.pth.tar --evaluate
```

| Method    | COCO    |VOC2007  |
|:---------:|:-------:|:-------:|
| Res-101 GAP  | 77.3    |   56.9   |
| Ours        |  83.0  | 62.5   |


## 3. Pre-trained models
will be added here.


## 4. Demo
will be added here.

## 5. Citation
will be added here.

### Acknowledgements
This code is based on ***Multi-Label Image Recognition with Graph Convolutional Networks*** ([Paper](https://arxiv.org/abs/1904.03582), [Code](https://github.com/Megvii-Nanjing/ML-GCN)). Fork of https://github.com/kprokofi/ML-GCN with updated code.


