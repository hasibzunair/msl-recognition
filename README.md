## Context Guided Image Recognition

[doc](https://docs.google.com/document/d/1yKBVNr90n2kipyQP4itzt3zvdUfGeSTm2qZ-MJNa8sg/edit?usp=sharing)


## 1. Specification of dependencies

This code requires Python 3.8.12 and CUDA 11.2. Run `conda env create -f env.yml` to install the required packages

## 2a. Training code

### Dataset details
We expect VOC2007, COCO2014 and Wider-Attribute dataset to have the following structure:
```
Dataset/
|-- VOCdevkit/
|---- VOC2007/
|------ JPEGImages/
|------ Annotations/
|------ ImageSets/
......
|-- COCO2014/
|---- annotations/
|---- images/
|------ train2014/
|------ val2014/
......
|-- WIDER/
|---- Annotations/
|------ wider_attribute_test.json
|------ wider_attribute_trainval.json
|---- Image/
|------ train/
|------ val/
|------ test/
...
```
Then directly run the following command to generate json file (for implementation) of these datasets.
```shell
python utils/prepare/prepare_voc.py  --data_path  datasets/VOCdevkit
python utils/prepare/prepare_coco.py --data_path  datasets/COCO2014
python utils/prepare/prepare_wider.py --data_path datasets/WIDER
```
which will automatically result in annotation json files in *./data/voc07*, *./data/coco* and *./data/wider*

### VOC 2007

## vit
`CUDA_VISIBLE_DEVICES=0 python main.py --exp_name vitl_voc --model vit_L16_224 --img_size 224 --batch_size 8 --num_heads 1 --lam 0.3 --dataset voc07 --num_cls 20`
## res
`CUDA_VISIBLE_DEVICES=0 python main.py --exp_name res_voc --num_heads 1 --lam 0.1 --dataset voc07 --num_cls 20`
## resnet cm
`CUDA_VISIBLE_DEVICES=0 python main.py --exp_name rescm_voc --batch_size 8 --num_heads 1 --lam 0.1 --dataset voc07 --num_cls 20 --cutmix data/resnet101_cutmix_pretrained.pth`

### COCO 2014
## vit
`CUDA_VISIBLE_DEVICES=0 python main.py --exp_name vitl_coco --model vit_L16_224 --img_size 224 --batch_size 8 --num_heads 8 --lam 0.3 --dataset coco --num_cls 80`
## resnet cm
`CUDA_VISIBLE_DEVICES=0 python main.py --exp_name rescm_coco --batch_size 8 --num_heads 6 --lam 0.4 --dataset coco --num_cls 80 --cutmix data/resnet101_cutmix_pretrained.pth`

## 2b. Evaluation code

### VOC 2007
## vit
`CUDA_VISIBLE_DEVICES=0 python val.py --model vit_L16_224 --img_size 224 --num_heads 1 --lam 0.3 --dataset voc07 --num_cls 20  --load_from checkpoint/vitl_voc/epoch_30.pth`
## resnet
`CUDA_VISIBLE_DEVICES=0 python val.py --num_heads 1 --lam 0.1 --dataset voc07 --num_cls 20  --load_from checkpoint/res_voc/epoch_30.pth`
## resnet cm
`CUDA_VISIBLE_DEVICES=0 python val.py --num_heads 1 --lam 0.1 --dataset voc07 --num_cls 20  --load_from checkpoint/rescm_voc/epoch_30.pth --cutmix data/resnet101_cutmix_pretrained.pth`

### COCO 2014
## vit
CUDA_VISIBLE_DEVICES=0 python val.py --model vit_L16_224 --img_size 224 --num_heads 8 --lam 0.3 --dataset coco --num_cls 80  --load_from checkpoint/vitl_coco/epoch_30.pth
## resnet cm
CUDA_VISIBLE_DEVICES=0 python val.py --num_heads 6 --lam 0.4 --dataset coco --num_cls 80  --load_from checkpoint/rescm_coco/epoch_30.pth --cutmix data/resnet101_cutmix_pretrained.pth


## 3. Pre-trained models
will be added here.


## 4. Demo
will be added here.

## 5. Citation
will be added here.

### Acknowledgements
This code is based on ***Residual Attention: A Simple But Effective Method for Multi-Label Recoginition*** ([Paper](https://arxiv.org/abs/2108.02456), [Code(https://github.com/Kevinz-code/CSRA)).

