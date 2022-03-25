## Object recognition with GCNs

[doc](https://docs.google.com/document/d/1yKBVNr90n2kipyQP4itzt3zvdUfGeSTm2qZ-MJNa8sg/edit?usp=sharing)

## 1. Specification of dependencies

This code requires Python 3.8.12. Install the following packages:

- numpy
- torchnet
- torch-1.8.2
- torchvision-0.9.2
- tqdm

## 2. Training and evaluation code

#### trainval on Pascal VOC 2007
```sh
python3 demo_voc2007_gcn.py data/voc --image-size 448 --batch-size 4
``` 

#### trainval on COCO 2014
```sh
python3 demo_coco_gcn.py data/coco --image-size 448 --batch-size 8
```

## 3. Pre-trained models
will be added here.


## 4. Demo
will be added here.


### Acknowledgements
This codebase is based on ***Multi-Label Image Recognition with Graph Convolutional Networks*** ([Paper](https://arxiv.org/abs/1904.03582), [Code](https://github.com/Megvii-Nanjing/ML-GCN)) 


