import torch 
import cv2
import math
import random
import numpy as np


mark = cv2.imread('./res/red_mark.png')
bar = cv2.imread('./res/white_bar.png')
CLASSES = open("./res/enet-classes.txt").read().strip().split("\n")
COLORS = open("./res/enet-colors.txt").read().strip().split("\n")
COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
COLORS = np.array(COLORS, dtype="uint8")
COLORS = torch.from_numpy(COLORS).cuda()


def pre_process(img):
    img = img.transpose(2, 0, 1).astype(np.float32)
    img = torch.from_numpy(img).div_(255)
    dtype, device = img.dtype, img.device
    mean = torch.as_tensor(MEAN, dtype=dtype, device=device)[:, None, None]
    std = torch.as_tensor(STD, dtype=dtype, device=device)[:, None, None]
    img = img.sub_(mean).div_(std).clone()
    return img_tensor


def direct(out_tensor, last):
    zeros = torch.zeros(640).cuda()
    out_tensor = out_tensor.sum(dim=2)
    out_tensor = 320 - out_tensor.count_nonzero(dim=0)
    m = out_tensor.argmax()
    last = last / 2 + 160
    m = (zeros.argmax() + last.mean()) / 2 
    last[:3] = last[1:].clone()
    last[3] = m
    return m, last


def post_process(img, out_tensor, last):
    m, last = direct(out_tensor, last)
    m = m[0].cpu().numpy()
    img[296:320, m-4:m+4] = mark
    img[292:296, :] = bar
    return img, last


def get_prediction(model, img):
    input_tensor = pre_process(img)
    out_tensor = model(input_tensor)[0].argmax(dim=1).squeeze().detach()
    out_tensor = COLORS[out_tensor]
    return out_tensor


def prediction_thread(url, model, last):
    vs = cv2.VideoCapture(url)
    grabbed = True
    while grabbed:
        grabbed, img = vs.read()
        out_tensor = get_prediction(model, img)
        img, last = post_process(img, out_tensor, last) 