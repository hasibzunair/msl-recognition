# preprocess data VOC2007 and COCO2014
python utils/prepare/prepare_voc.py  --data_path  datasets/VOCdevkit
python utils/prepare/prepare_coco.py --data_path  datasets/COCO2014

# VOC train
## vit
CUDA_VISIBLE_DEVICES=0 python main.py --exp_name vitl_voc --model vit_L16_224 --img_size 224 --batch_size 8 --num_heads 1 --lam 0.3 --dataset voc07 --num_cls 20
## res
CUDA_VISIBLE_DEVICES=0 python main.py --exp_name res_voc --num_heads 1 --lam 0.1 --dataset voc07 --num_cls 20
## resnet cm
CUDA_VISIBLE_DEVICES=0 python main.py --exp_name rescm_voc --batch_size 8 --num_heads 1 --lam 0.1 --dataset voc07 --num_cls 20 --cutmix data/resnet101_cutmix_pretrained.pth

# VOC val
## vit
CUDA_VISIBLE_DEVICES=0 python val.py --model vit_L16_224 --img_size 224 --num_heads 1 --lam 0.3 --dataset voc07 --num_cls 20  --load_from checkpoint/vitl_voc/epoch_30.pth
## resnet
CUDA_VISIBLE_DEVICES=0 python val.py --num_heads 1 --lam 0.1 --dataset voc07 --num_cls 20  --load_from checkpoint/res_voc/epoch_30.pth
## resnet cm
CUDA_VISIBLE_DEVICES=0 python val.py --num_heads 1 --lam 0.1 --dataset voc07 --num_cls 20  --load_from checkpoint/rescm_voc/epoch_30.pth --cutmix data/resnet101_cutmix_pretrained.pth


# COCO train
## vit
CUDA_VISIBLE_DEVICES=0 python main.py --exp_name vitl_coco --model vit_L16_224 --img_size 224 --batch_size 8 --num_heads 8 --lam 0.3 --dataset coco --num_cls 80
## resnet cm
CUDA_VISIBLE_DEVICES=0 python main.py --exp_name rescm_coco --batch_size 8 --num_heads 6 --lam 0.4 --dataset coco --num_cls 80 --cutmix data/resnet101_cutmix_pretrained.pth

# COCO val
## vit
CUDA_VISIBLE_DEVICES=0 python val.py --model vit_L16_224 --img_size 224 --num_heads 8 --lam 0.3 --dataset coco --num_cls 80  --load_from checkpoint/vitl_coco/epoch_30.pth
## resnet cm
CUDA_VISIBLE_DEVICES=0 python val.py --num_heads 6 --lam 0.4 --dataset coco --num_cls 80  --load_from checkpoint/rescm_coco/epoch_30.pth --cutmix data/resnet101_cutmix_pretrained.pth


# todo
- do cool stuff