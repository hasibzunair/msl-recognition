import os
import sys
import torch
import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms

sys.path.insert(0, "../")
from pipeline.resnet_csra import ResNet_CSRA
from utils.evaluation.eval import class_dict

torch.manual_seed(0)

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

# Use GPU if available
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# Make directories
os.system("mkdir ./temp")

# Get model weights
if not os.path.exists("./temp/msl_c_voc.pth"):
    os.system(
        "wget -O ./temp/msl_c_voc.pth https://github.com/hasibzunair/msl-recognition/releases/download/v1.0-models/msl_c_voc.pth"
    )
if not os.path.exists("./temp/msl_c_coco.pth"):
    os.system(
        "wget -O ./temp/msl_c_coco.pth https://github.com/hasibzunair/msl-recognition/releases/download/v1.0-models/msl_c_coco.pth"
    )


### Tests

def test_model_pretrained():
    
    # VOC
    model = ResNet_CSRA(num_heads=1, lam=0.1, num_classes=20)
    model.load_state_dict(torch.load("./temp/msl_c_voc.pth", map_location=torch.device("cpu")))
    
    # COCO
    model = ResNet_CSRA(num_heads=6, lam=0.4, num_classes=80)
    model.load_state_dict(torch.load("./temp/msl_c_coco.pth", map_location=torch.device("cpu")))


def test_model_function():

    # load VOC pretrained model
    model = ResNet_CSRA(num_heads=1, lam=0.1, num_classes=20)
    model.load_state_dict(torch.load("./temp/msl_c_voc.pth", map_location=torch.device("cpu")))
    model.to(DEVICE)
    model.eval()

    # read image
    image = Image.open("./tests/000001.jpg").convert("RGB")

    # image pre-process
    normalize = transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
    transforms_image = transforms.Compose(
        [transforms.Resize((448, 448)), transforms.ToTensor(), normalize]
    )
    image = transforms_image(image)
    image = image.unsqueeze(0)

    # predict
    result = []
    with torch.no_grad():
        image = image.to(DEVICE)
        logit = model(image).squeeze(0)
        logit = nn.Sigmoid()(logit)

        pos = torch.where(logit > 0.5)[0].cpu().numpy()
        for k in pos:
            result.append(str(class_dict["voc07"][k]))
    assert result != None, f"Should not be empty, got: {result} which is {type(result)}"


if __name__ == "__main__":
    test_model_pretrained()
    test_model_function()