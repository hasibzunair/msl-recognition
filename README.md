## MaskSup for Multi-label Image Recognition

[doc](https://docs.google.com/document/d/1yKBVNr90n2kipyQP4itzt3zvdUfGeSTm2qZ-MJNa8sg/edit?usp=sharing)


## 1. Specification of dependencies

This code requires Python 3.8.12 and CUDA 11.2. Create and activate the following conda envrionment.

```
conda update conda
conda env create -f environment.yml
conda activate maskrec
```

## 2a. Training code

### Dataset details
We expect VOC2007, COCO2014 and Wider-Attribute dataset to have the following structure:
```
datasets/
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

### VOC2007 training

For Baseline ResNet with CutMix:

```shell
CUDA_VISIBLE_DEVICES=0 python train.py --exp_name base_rescm_voc --batch_size 6 --total_epoch 60 --num_heads 1 --lam 0.1 --dataset voc07 --num_cls 20 --cutmix data/resnet101_cutmix_pretrained.pth
```

For MSL ResNet with CutMix: 

```shell
CUDA_VISIBLE_DEVICES=0 python train_masksup.py --exp_name masksup_rescm_voc --batch_size 6 --total_epoch 60 --num_heads 1 --lam 0.1 --dataset voc07 --num_cls 20 --cutmix data/resnet101_cutmix_pretrained.pth
```

For Baseline ViT
```
CUDA_VISIBLE_DEVICES=0 python train.py --exp_name vitl_voc_base --model vit_L16_224 --img_size 224 --batch_size 6 --total_epoch 60 --num_heads 1 --lam 0.3 --dataset voc07 --num_cls 20
```

For MSL ViT
```
CUDA_VISIBLE_DEVICES=0 python train_masksup.py --exp_name masksup_vitl_voc --model vit_L16_224 --img_size 224 --batch_size 6 --total_epoch 60 --num_heads 1 --lam 0.3 --dataset voc07 --num_cls 20
```

### MS-COCO training

For Baseline ResNet with CutMix:

```shell
CUDA_VISIBLE_DEVICES=0 python train.py --exp_name rescm_paper_coco --batch_size 6 --total_epoch 60 --num_heads 6 --lam 0.4 --dataset coco --num_cls 80 --cutmix data/resnet101_cutmix_pretrained.pth
```

For MSL ResNet with CutMix: 

```
CUDA_VISIBLE_DEVICES=0 python train_masksup.py --exp_name masksup01_0.3,0.2,0.5_rescm_coco --batch_size 6 --total_epoch 60 --num_heads 6 --lam 0.4 --dataset coco --num_cls 80 --cutmix data/resnet101_cutmix_pretrained.pth
```

For Baseline ViT
```
CUDA_VISIBLE_DEVICES=0 python train.py --exp_name vitl_coco --model vit_L16_224 --img_size 224 --batch_size 6 --total_epoch 40 --num_heads 8 --lam 1 --dataset coco --num_cls 80
```

For MSL ViT
```
CUDA_VISIBLE_DEVICES=0 python train_masksup.py --exp_name masksup_vitl_coco --model vit_L16_224 --img_size 224 --batch_size 6 --total_epoch 40 --num_heads 8 --lam 1 --dataset coco --num_cls 80
```
### WIDER-Attribute training

For ViT
```
CUDA_VISIBLE_DEVICES=0 python train.py --exp_name vitl_wider --model vit_L16_224 --img_size 224 --batch_size 6 --total_epoch 40 --num_heads 1 --lam 0.3 --dataset wider --num_cls 14
```

For MSL ViT
```
CUDA_VISIBLE_DEVICES=0 python train_masksup.py --exp_name masksup_vitl_wider --model vit_L16_224 --img_size 224 --batch_size 6 --total_epoch 40 --num_heads 1 --lam 0.3 --dataset wider --num_cls 14
```

### Tresnet

git+https://github.com/mapillary/inplace_abn.git@v1.0.12

```
CUDA_VISIBLE_DEVICES=0 python train.py --exp_name tresnet_voc --model tresnet_m --batch_size 6 --total_epoch 60 --dataset voc07 --num_cls 20 --tres ./data/tresnet_m_448.pth
```

## 2b. Evaluation code

### VOC2007

For Baseline ResNet with CutMix:
```shell
CUDA_VISIBLE_DEVICES=0 python val.py --num_heads 1 --lam 0.1 --dataset voc07 --num_cls 20  --load_from checkpoint/voc_experiments/rescm_paper_voc/epoch_200.pth --cutmix data/resnet101_cutmix_pretrained.pth
```

### COCO2014

For Baseline ResNet with CutMix
```shell 
CUDA_VISIBLE_DEVICES=0 python val.py --num_heads 6 --lam 0.4 --dataset coco --num_cls 80  --load_from checkpoint/rescm_paper_coco/epoch_100.pth --cutmix data/resnet101_cutmix_pretrained.pth
```

## 3. Pre-trained models

TBA.

## 4. Demo

We provide prediction demos of our models. The demo images (picked from VCO2007) have already been put into *./utils/demo_images/*, you can simply run demo.py by using our CSRA models pretrained on VOC2007:
```shell
CUDA_VISIBLE_DEVICES=0 python demo.py --model resnet101 --num_heads 1 --lam 0.1 --dataset voc07 --load_from PATH_TO_YOUR_RESNET_MODEL --img_dir utils/demo_images
```
which will output like this:
```shell
utils/demo_images/000001.jpg prediction: dog,person,
utils/demo_images/000004.jpg prediction: car,
utils/demo_images/000002.jpg prediction: train,
...
```

## 5. Citation

TBA.

### Acknowledgements

This code is based on ***Residual Attention: A Simple But Effective Method for Multi-Label Recoginition*** ([Paper](https://arxiv.org/abs/2108.02456), [Code](https://github.com/Kevinz-code/CSRA)). We thank authors for making their code public! :D 

