# Code for MSL
# Author: Hasib Zunair

import argparse
import time
import random
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from tqdm import tqdm

from pipeline.resnet_csra import ResNet_CSRA
from pipeline.vit_csra import VIT_B16_224_CSRA, VIT_L16_224_CSRA, VIT_CSRA
from pipeline.dataset import DataSetMaskSup
from utils.evaluation.eval import evaluation
from utils.evaluation.warmUpLR import WarmUpLR
from helpers import Logger


def get_argparser():
    parser = argparse.ArgumentParser(description="settings")
    # configuration
    parser.add_argument("--exp_name", default="baseline")
    # model
    parser.add_argument("--model", default="resnet101")
    parser.add_argument("--num_heads", default=1, type=int)
    parser.add_argument("--lam", default=0.1, type=float)
    parser.add_argument(
        "--cutmix", default=None, type=str
    )  # path to cutmix-pretrained backbone
    parser.add_argument(
        "--tres", default=None, type=str
    )  # path to tresnet-pretrained backbone
    # dataset
    parser.add_argument("--dataset", default="voc07", type=str)
    parser.add_argument("--num_cls", default=20, type=int)
    parser.add_argument("--train_aug", default=["randomflip", "resizedcrop"], type=list)
    parser.add_argument("--test_aug", default=[], type=list)
    parser.add_argument("--img_size", default=448, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    # optimizer, default SGD
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--w_d", default=0.0001, type=float, help="weight_decay")
    parser.add_argument("--warmup_epoch", default=2, type=int)
    parser.add_argument("--total_epoch", default=30, type=int)
    parser.add_argument("--print_freq", default=100, type=int)
    args = parser.parse_args()
    return args


def train_masksup(i, args, model, train_loader, optimizer, warmup_scheduler):

    print("Starting training...")
    model.train()
    epoch_begin = time.time()

    for index, data in enumerate(train_loader):
        batch_begin = time.time()

        # Get image, masked image and label
        img = data["img"].cuda()
        masked_img = data["masked_img"].cuda()
        target = data["target"].cuda()

        #### Compute loss ####
        optimizer.zero_grad()

        # Original branch loss
        logit1, loss1 = model(img, target)
        loss1 = loss1.mean()

        # Context branch loss
        logit2, loss2 = model(masked_img, target)
        loss2 = loss2.mean()

        # Task similarity loss
        pred1 = torch.sigmoid(logit1.float())
        pred2 = torch.sigmoid(logit2.float())
        loss3 = criterion_mse(pred1, pred2)

        # Loss coefficients
        alpha = 0.3
        beta = 0.2
        gamma = 0.5

        # Compute total loss
        loss = alpha * loss1 + beta * loss2 + gamma * loss3

        #### Update ####
        loss.backward()
        optimizer.step()

        t = time.time() - batch_begin
        if index % args.print_freq == 0:
            print(
                "Epoch {}[{}/{}]: loss:{:.5f}, lr:{:.5f}, time:{:.4f}".format(
                    i,
                    args.batch_size * (index + 1),
                    len(train_loader.dataset),
                    loss,
                    optimizer.param_groups[0]["lr"],
                    float(t),
                )
            )

        if warmup_scheduler and i <= args.warmup_epoch:
            warmup_scheduler.step()

    t = time.time() - epoch_begin
    print("Epoch {} training ends, total {:.2f}s".format(i, t))


def val(i, args, model, test_loader, test_file):
    model.eval()
    print("Test on Epoch {}".format(i))
    result_list = []

    for index, data in enumerate(tqdm(test_loader)):

        # Get image and label
        img = data["img"].cuda()
        target = data["target"].cuda()
        img_path = data["img_path"]

        # Get predictions
        with torch.no_grad():
            logit = model(img)

        result = nn.Sigmoid()(logit).cpu().detach().numpy().tolist()

        for k in range(len(img_path)):
            result_list.append(
                {
                    "file_name": img_path[k].split("/")[-1].split(".")[0],
                    "scores": result[k],
                }
            )
    # cal_mAP OP OR
    evaluation(result=result_list, types=args.dataset, ann_path=test_file[0])


def main():

    ########## Reproducibility ##########
    random.seed(0)
    np.random.seed(0)
    os.environ["PYTHONHASHSEED"] = str(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    ########## Get arguments ##########
    args = get_argparser()

    # create log folder
    if not os.path.exists("checkpoint/"):
        os.mkdir("checkpoint/")
    if not os.path.exists("checkpoint/" + args.exp_name):
        os.mkdir("checkpoint/" + args.exp_name)

    # save config in log file
    sys.stdout = Logger(os.path.join("checkpoint", args.exp_name, "log_train.txt"))
    # print("=========================\nConfigs:{}\n=========================".format(args))
    s = str(args).split(", ")
    print("=========================\nConfigs:{}\n=========================")
    for i in range(len(s)):
        print(s[i])
    print("=========================")

    ########## Define model ##########
    if args.model == "resnet101":
        model = ResNet_CSRA(
            num_heads=args.num_heads,
            lam=args.lam,
            num_classes=args.num_cls,
            cutmix=args.cutmix,
        )
    if args.model == "vit_B16_224":
        model = VIT_B16_224_CSRA(
            cls_num_heads=args.num_heads, lam=args.lam, cls_num_cls=args.num_cls
        )
    if args.model == "vit_L16_224":
        model = VIT_L16_224_CSRA(
            cls_num_heads=args.num_heads, lam=args.lam, cls_num_cls=args.num_cls
        )

    # Send model to GPU
    model.cuda()
    if torch.cuda.device_count() > 1:
        print("lets use {} GPUs.".format(torch.cuda.device_count()))
        model = nn.DataParallel(
            model, device_ids=list(range(torch.cuda.device_count()))
        )

    ########## Load data ##########
    if args.dataset == "voc07":
        train_file = ["data/voc07/trainval_voc07.json"]
        test_file = ["data/voc07/test_voc07.json"]
        step_size = 4
    if args.dataset == "coco":
        train_file = ["data/coco/train_coco2014.json"]
        test_file = ["data/coco/val_coco2014.json"]
        step_size = 5
    if args.dataset == "wider":
        train_file = ["data/wider/trainval_wider.json"]
        test_file = ["data/wider/test_wider.json"]
        step_size = 5
        args.train_aug = ["randomflip"]

    train_dataset = DataSetMaskSup(
        train_file, args.train_aug, args.img_size, args.dataset
    )
    test_dataset = DataSetMaskSup(test_file, args.test_aug, args.img_size, args.dataset)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8
    )

    ########## Setup loss, optimizer and warmup ##########
    global criterion_mse
    criterion_mse = nn.MSELoss()

    backbone, classifier = [], []
    for name, param in model.named_parameters():
        if "classifier" in name:
            classifier.append(param)
        else:
            backbone.append(param)
    optimizer = optim.SGD(
        [
            {"params": backbone, "lr": args.lr},
            {"params": classifier, "lr": args.lr * 10},
        ],
        momentum=args.momentum,
        weight_decay=args.w_d,
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

    iter_per_epoch = len(train_loader)
    if args.warmup_epoch > 0:
        warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warmup_epoch)
    else:
        warmup_scheduler = None

    ########## Training and evaluation loop ##########
    for i in range(1, args.total_epoch + 1):
        train_masksup(i, args, model, train_loader, optimizer, warmup_scheduler)
        torch.save(
            model.state_dict(), "checkpoint/{}/epoch_{}.pth".format(args.exp_name, i)
        )
        val(i, args, model, test_loader, test_file)
        scheduler.step()


if __name__ == "__main__":
    main()
