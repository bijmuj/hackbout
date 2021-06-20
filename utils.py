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
MEAN=(0.3257, 0.3690, 0.3223) # city, rgb
STD=(0.2112, 0.2148, 0.2115)



def pre_process(img):
    input_tensor = img.transpose(2, 0, 1).astype(np.float32)
    input_tensor = torch.from_numpy(input_tensor).div_(255).cuda()
    dtype, device = input_tensor.dtype, input_tensor.device
    mean = torch.as_tensor(MEAN, dtype=dtype, device=device)[:, None, None]
    std = torch.as_tensor(STD, dtype=dtype, device=device)[:, None, None]
    input_tensor = input_tensor.sub_(mean).div_(std).clone()
    return input_tensor.unsqueeze(0)


def direct(out_tensor, last):
    zeros = torch.zeros(640).cuda()
    out_tensor = out_tensor.sum(dim=2)
    out_tensor = 320 - torch.count_nonzero(out_tensor, dim=0)
    m = out_tensor.argmax()
    last = last / 2 + 160
    m = (m + last.mean()) / 2 
    new_last = last.clone()
    new_last[:3] = last[1:]
    new_last[-1] = m
    return m, new_last


def post_process(img, out_tensor, last):
    m, last = direct(out_tensor, last)
    m = int(m.cpu().item())
    img[296:320, m-4:m+4] = mark
    img[292:296, :] = bar
    return img, last


def get_prediction(model, input_tensor):
    out_tensor = model(input_tensor)[0].argmax(dim=1).squeeze().detach()
    out_tensor = COLORS[out_tensor]
    return out_tensor


