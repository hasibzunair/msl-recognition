python trainval_glas_polyp.py --exp_name glas_bin_levit
python trainval_glas_polyp.py --exp_name polyp_bin_levit
python trainval_nyudv2.py --exp_name nyu_bin_nestunet

# https://arxiv.org/abs/2210.00923
# Method GLaS, mIoU (↑) CVC-Clinic-DB, mIoU (↑) NYUDv2 (↑)
# MAE [8] 75.04 82.50 37.42
# MaskSup (BMVC) 76.06 84.02 39.31
# 