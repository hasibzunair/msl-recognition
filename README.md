# MSL

**Concordia University**

Hasib Zunair and A. Ben Hamza

[[`Paper`](link)] [[`Project`](link)] [[`Demo`](#4-demo)] [[`BibTeX`](#5-citation)]

This is official code for our **WACV 2024 paper**:<br>
[Learning to Recognize Occluded and Small Objects with Partial Inputs](Link)
<br>

![MSL Design](https://github.com/hasibzunair/masksup-segmentation/blob/master/media/pipeline.png?raw=true)

Summarize in 3-5 sentences your project here.

## 1. Specification of dependencies

This code requires Python 3.8.12 and CUDA 11.2. Create and activate the following conda envrionment.

```bash
conda update conda
conda env create -f environment.yml
conda activate msl
```

## 2a. Training code

### Dataset details

The VOC2007, COCO2014 and Wider-Attribute datasets are expected to have the following structure:

```bash
|- datasets/
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

```bash
python utils/prepare/prepare_voc.py  --data_path  datasets/VOCdevkit
python utils/prepare/prepare_coco.py --data_path  datasets/COCO2014
python utils/prepare/prepare_wider.py --data_path datasets/WIDER
```

which will automatically result in annotation json files in *./data/voc07*, *./data/coco* and *./data/wider*

### VOC2007 training

```bash
# MSL ResNet with CutMix
CUDA_VISIBLE_DEVICES=0 python train.py --exp_name msl_rescm_voc --batch_size 6 --total_epoch 60 --num_heads 1 --lam 0.1 --dataset voc07 --num_cls 20 --cutmix data/resnet101_cutmix_pretrained.pth
```

```bash
# MSL ViT
CUDA_VISIBLE_DEVICES=0 python train.py --exp_name msl_vitl_voc --model vit_L16_224 --img_size 224 --batch_size 6 --total_epoch 60 --num_heads 1 --lam 0.3 --dataset voc07 --num_cls 20
```

### MS-COCO training

```bash
# MSL ResNet with CutMix
CUDA_VISIBLE_DEVICES=0 python train.py --exp_name msl01_0.3,0.2,0.5_rescm_coco --batch_size 6 --total_epoch 60 --num_heads 6 --lam 0.4 --dataset coco --num_cls 80 --cutmix data/resnet101_cutmix_pretrained.pth
```

```bash
# MSL ViT
CUDA_VISIBLE_DEVICES=0 python train.py --exp_name msl_vitl_coco --model vit_L16_224 --img_size 224 --batch_size 6 --total_epoch 40 --num_heads 8 --lam 1 --dataset coco --num_cls 80
```

### WIDER-Attribute training

```bash
# MSL ViT-L
CUDA_VISIBLE_DEVICES=0 python train.py --exp_name msl_vitl_wider --model vit_L16_224 --img_size 224 --batch_size 6 --total_epoch 40 --num_heads 1 --lam 0.3 --dataset wider --num_cls 14
```

```bash
# MSL ViT-B
CUDA_VISIBLE_DEVICES=0 python train.py --exp_name msl_vitb_wider --model vit_B16_224 --img_size 224 --batch_size 6 --total_epoch 40 --num_heads 1 --lam 0.3 --dataset wider --num_cls 14
```

## 2b. Evaluation code

### VOC2007

```bash
# MSL ResNet with CutMix
CUDA_VISIBLE_DEVICES=0 python val.py --num_heads 1 --lam 0.1 --dataset voc07 --num_cls 20  --load_from checkpoint/msl_c_voc.pth
```

### COCO2014

```bash
# MSL ResNet with CutMix
CUDA_VISIBLE_DEVICES=0 python val.py --num_heads 6 --lam 0.4 --dataset coco --num_cls 80  --load_from checkpoint/msl_c_coco.pth
```

### Wider-Attribute
```bash
CUDA_VISIBLE_DEVICES=0 python val.py --model vit_B16_224 --img_size 224 --num_heads 1 --lam 0.3 --dataset wider --num_cls 14  --load_from checkpoint/msl_v_wider.pth
```

All experiments are conducted on a single NVIDIA 3080Ti GPU. For additional implementation details and results, please refer to the supplementary material [here](https://github.com/hasibzunair/masksup-segmentation/blob/master/media/supplementary_materials.pdf).

## 3. Pre-trained models

We provide pretrained models on [GitHub Releases](https://github.com/hasibzunair/masksup-segmentation/releases/tag/v0.1) for reproducibility.

|Dataset      | Backbone  |   mAP (%)  |   Download   |
|  ---------- | -------   |  ------ |  --------   |
| VOC2007 | MSL-C  | 86.4 | [download](https://github.com/hasibzunair/masksup-segmentation/releases/download/v0.1/masksupglas76.06iou.pth) |
| COCO2014 | MSL-C | 96.1 | [download](https://github.com/hasibzunair/masksup-segmentation/releases/download/v0.1/masksuppolyp84.02iou.pth) |
| Wider-Attribute | MSL-V | 90.6 | [download](https://github.com/hasibzunair/masksup-segmentation/releases/download/v0.1/masksupnyu39.31iou.pth) |

## 4. Demo

We provide prediction demos of our models. The demo images (picked from VCO2007) have already been put into *./utils/demo_images/*, you can simply run demo.py by using our MSL models pretrained on VOC2007:

```bash
CUDA_VISIBLE_DEVICES=0 python demo.py --model resnet101 --dataset voc07 --load_from checkpoint/msl_c_voc.pth --img_dir utils/demo_images
```

which will output like this:

```bash
utils/demo_images/000001.jpg prediction: dog,person,
utils/demo_images/000004.jpg prediction: car,
utils/demo_images/000002.jpg prediction: train,
...
```

## 5. Citation

```bibtex
 @inproceedings{zunair2024msl,
    title={Learning to Recognize Occluded and Small Objects with Partial Inputs},
    author={Zunair, Hasib and Hamza, A Ben},
    booktitle={Proc. IEEE Winter Conference on Applications of Computer Vision},
    year={2024}
  }
```

### Acknowledgements

This code is based on ***Residual Attention: A Simple But Effective Method for Multi-Label Recoginition*** ([Paper](https://arxiv.org/abs/2108.02456), [Code](https://github.com/Kevinz-code/CSRA)). We thank authors for making their code public.