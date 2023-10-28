import os
import torch
import gradio as gr
import argparse
import codecs
import time
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms

from pipeline.resnet_csra import ResNet_CSRA
from pipeline.vit_csra import VIT_B16_224_CSRA, VIT_L16_224_CSRA, VIT_CSRA
from pipeline.dataset import DataSet
from torchvision.transforms import transforms
from utils.evaluation.eval import voc_classes, wider_classes, coco_classes, class_dict

torch.manual_seed(0)

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

# Device
# Use GPU if available
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# Make directories
os.system("mkdir ./models")

# Get model weights
if not os.path.exists("./models/msl_c_voc.pth"):
    os.system(
        "wget -O ./models/msl_c_voc.pth https://github.com/hasibzunair/msl-recognition/releases/download/v1.0-models/msl_c_voc.pth"
    )

# Load model
model = ResNet_CSRA(num_heads=1, lam=0.1, num_classes=20)
normalize = transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
model.to(DEVICE)
print("Loading weights from {}".format("./models/msl_c_voc.pth"))
model.load_state_dict(torch.load("./models/msl_c_voc.pth", map_location=torch.device("cpu")))
model.eval()

# Inference!
def inference(img_path):
    # read image
    image = Image.open(img_path).convert("RGB")

    # image pre-process
    transforms_image = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        normalize
    ])

    image = transforms_image(image)
    image = image.unsqueeze(0)

    # Predict
    result = []
    with torch.no_grad():
        image = image.to(DEVICE)
        logit = model(image).squeeze(0)
        logit = nn.Sigmoid()(logit)

        pos = torch.where(logit > 0.5)[0].cpu().numpy()
        for k in pos:
            result.append(str(class_dict["voc07"][k]))
    return result


# Define ins outs placeholders
inputs = gr.inputs.Image(type="filepath", label="Input Image")

# Define style
title = "Learning to Recognize Occluded and Small Objects with Partial Inputs"
description = codecs.open("description.html", "r", "utf-8").read()
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/1512.03385' target='_blank'>Learning to Recognize Occluded and Small Objects with Partial Inputs</a> | <a href='https://github.com/hasibzunair/msl-recognition' target='_blank'>Github Repo</a></p>"

voc_classes = ("aeroplane", "bicycle", "bird", "boat", "bottle",
           "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor")

# Run inference
gr.Interface(inference, 
            inputs, 
            outputs="text", 
            examples=["./000001.jpg", "./000006.jpg", "./000009.jpg"], 
            title=title, 
            description=description, 
            article=article,
            analytics_enabled=False).launch()
