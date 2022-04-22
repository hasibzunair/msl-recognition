## Context Guided Image Recognition

[doc](https://docs.google.com/document/d/1yKBVNr90n2kipyQP4itzt3zvdUfGeSTm2qZ-MJNa8sg/edit?usp=sharing)


## 1. Specification of dependencies

This code requires Python 3.8.12. Run `conda env create -f env.yml` to install the required packages

## 2a. Training code

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


## 3. Pre-trained models
will be added here.


## 4. Demo
will be added here.

## 5. Citation
will be added here.

### Acknowledgements
This code is based on ***Residual Attention: A Simple But Effective Method for Multi-Label Recoginition*** ([Paper](https://arxiv.org/abs/2108.02456), [Code(https://github.com/Kevinz-code/CSRA)).

