import json
import glob
import random 

from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import transforms
import torch
import numpy as np

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


# modify for transformation for vit
# modfify wider crop-person images


###### Base data loader ######
# class DataSet(Dataset):
#     def __init__(
#         self,
#         ann_files,
#         augs,
#         img_size,
#         dataset,
#     ):
#         self.dataset = dataset
#         self.ann_files = ann_files
#         self.augment = self.augs_function(augs, img_size)
#         self.transform = transforms.Compose(
#             [transforms.ToTensor(), transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])]
#             # In this paper, we normalize the image data to [0, 1]
#             # You can also use the so called 'ImageNet' Normalization method
#         )
#         self.anns = []
#         self.load_anns()
#         print(self.augment)

#         # in wider dataset we use vit models
#         # so transformation has been changed
#         if self.dataset == "wider":
#             self.transform = transforms.Compose(
#                 [
#                     transforms.ToTensor(),
#                     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
#                 ]
#             )

#     def augs_function(self, augs, img_size):
#         t = []
#         if "randomflip" in augs:
#             t.append(transforms.RandomHorizontalFlip())
#         if "ColorJitter" in augs:
#             t.append(
#                 transforms.ColorJitter(
#                     brightness=0.5, contrast=0.5, saturation=0.5, hue=0
#                 )
#             )
#         if "resizedcrop" in augs:
#             t.append(transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)))
#         if "RandAugment" in augs:
#             t.append(RandAugment())

#         t.append(transforms.Resize((img_size, img_size)))

#         return transforms.Compose(t)

#     def load_anns(self):
#         self.anns = []
#         for ann_file in self.ann_files:
#             json_data = json.load(open(ann_file, "r"))
#             self.anns += json_data

#     def __len__(self):
#         return len(self.anns)

#     def __getitem__(self, idx):
#         idx = idx % len(self)
#         ann = self.anns[idx]
#         img = Image.open(ann["img_path"]).convert("RGB")

#         if self.dataset == "wider":
#             x, y, w, h = ann["bbox"]
#             img_area = img.crop([x, y, x + w, y + h])
#             img_area = self.augment(img_area)
#             img_area = self.transform(img_area)
#             message = {
#                 "img_path": ann["img_path"],
#                 "target": torch.Tensor(ann["target"]),
#                 "img": img_area,
#             }
#         else:  # voc and coco
#             img = self.augment(img)
#             img = self.transform(img)
#             message = {
#                 "img_path": ann["img_path"],
#                 "target": torch.Tensor(ann["target"]),
#                 "img": img,
#             }

#         return message
#         # finally, if we use dataloader to get the data, we will get
#         # {
#         #     "img_path": list, # length = batch_size
#         #     "target": Tensor, # shape: batch_size * num_classes
#         #     "img": Tensor, # shape: batch_size * 3 * 224 * 224
#         # }


def preprocess_scribble(img):
    transform = transforms.Compose(
        [
            transforms.Resize(448, BICUBIC),
            transforms.CenterCrop(448),
            #_convert_image_to_rgb,
            transforms.ToTensor(),
        ]
    )
    return transform(img)


class DataSetMaskSup(Dataset):
    """
    Data loader with scribbles.
    """
    def __init__(
        self,
        ann_files,
        augs,
        img_size,
        dataset,
    ):
        self.dataset = dataset
        self.ann_files = ann_files
        self.augment = self.augs_function(augs, img_size)
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])]
            # In this paper, we normalize the image data to [0, 1]
            # You can also use the so called 'ImageNet' Normalization method
        )
        self.anns = []
        self.load_anns()
        print(self.augment)

        # scribbles
        self._scribbles_folder = "./datasets/SCRIBBLES"
        self._scribbles = sorted(glob.glob(self._scribbles_folder + "/*.png"))[::-1][
            :1000
        ]  # for heavy masking [::-1]

        # in wider dataset we use vit models
        # so transformation has been changed
        if self.dataset == "wider":
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )

    def augs_function(self, augs, img_size):
        t = []
        if "randomflip" in augs:
            t.append(transforms.RandomHorizontalFlip())
        if "ColorJitter" in augs:
            t.append(
                transforms.ColorJitter(
                    brightness=0.5, contrast=0.5, saturation=0.5, hue=0
                )
            )
        if "resizedcrop" in augs:
            t.append(transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)))
        if "RandAugment" in augs:
            t.append(RandAugment())

        t.append(transforms.Resize((img_size, img_size)))

        return transforms.Compose(t)

    def load_anns(self):
        self.anns = []
        for ann_file in self.ann_files:
            json_data = json.load(open(ann_file, "r"))
            self.anns += json_data

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        idx = idx % len(self)
        ann = self.anns[idx]
        img = Image.open(ann["img_path"]).convert("RGB")

        # get scribble
        scribble_path = self._scribbles[
                random.randint(0, 950)
            ]
        scribble = Image.open(scribble_path).convert('P')
        scribble = preprocess_scribble(scribble)
        
        # todo, try without this
        scribble = (scribble > 0).float() # threshold to [0,1]
        inv_scribble = (torch.max(scribble) - scribble) # inverted scribble

        if self.dataset == "wider":
            x, y, w, h = ann["bbox"]
            img_area = img.crop([x, y, x + w, y + h])
            img_area = self.augment(img_area)
            img_area = self.transform(img_area)

            # masked image
            masked_image = img_area * inv_scribble

            message = {
                "img_path": ann["img_path"],
                "target": torch.Tensor(ann["target"]),
                "img": img_area,
                "masked_img": masked_image,
                #"scribble": scribble,
            }
        else:  # voc and coco
            img = self.augment(img)
            img = self.transform(img)
            # masked image
            masked_image = img * inv_scribble
            message = {
                "img_path": ann["img_path"],
                "target": torch.Tensor(ann["target"]),
                "img": img,
                "masked_img": masked_image,
                #"scribble": scribble,
            }

        return message
        # finally, if we use dataloader to get the data, we will get
        # {
        #     "img_path": list, # length = batch_size
        #     "target": Tensor, # shape: batch_size * num_classes
        #     "img": Tensor, # shape: batch_size * 3 * 224 * 224
        #     "masked_img": Tensor, # shape: batch_size * 3 * 224 * 224
        # }
