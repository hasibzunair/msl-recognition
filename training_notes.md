# MSL Experiments

Scripts to train baselines and MSL models.

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

For Baseline TResNet

```
CUDA_VISIBLE_DEVICES=0 python train.py --exp_name tresnetxl_voc --model tresnet_xl --batch_size 6 --total_epoch 60 --dataset voc07 --num_cls 20 --tres ./data/tresnet_xl_448.pth
```

For MSL TResNet

```
CUDA_VISIBLE_DEVICES=0 python train_masksup.py --exp_name masksup_tresnetm_voc --model tresnet_m --batch_size 6 --total_epoch 60 --dataset voc07 --num_cls 20 --tres ./data/tresnet_m_448.pth
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

For ViT-L
```
CUDA_VISIBLE_DEVICES=0 python train.py --exp_name vitl_wider --model vit_L16_224 --img_size 224 --batch_size 6 --total_epoch 40 --num_heads 1 --lam 0.3 --dataset wider --num_cls 14
```

For MSL ViT-L
```
CUDA_VISIBLE_DEVICES=0 python train_masksup.py --exp_name masksup_vitl_wider --model vit_L16_224 --img_size 224 --batch_size 6 --total_epoch 40 --num_heads 1 --lam 0.3 --dataset wider --num_cls 14
```

For ViT-B
```
CUDA_VISIBLE_DEVICES=0 python train.py --exp_name vitb_wider --model vit_B16_224 --img_size 224 --batch_size 6 --total_epoch 40 --num_heads 1 --lam 0.3 --dataset wider --num_cls 14
```

For MSL ViT-B
```
CUDA_VISIBLE_DEVICES=0 python train_masksup.py --exp_name masksup_vitb_wider --model vit_B16_224 --img_size 224 --batch_size 6 --total_epoch 40 --num_heads 1 --lam 0.3 --dataset wider --num_cls 14
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
