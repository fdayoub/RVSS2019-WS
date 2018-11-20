#!/usr/bin/env python3
import time
import math
import cv2
import numpy as np
import torch
from glob import glob
from torch import nn
from steerNet import SteerNet
from torchvision import transforms


def load_net(wegihts_filename):
    model = SteerNet()
    model.load_state_dict(torch.load(wegihts_filename))
    model.eval()        
    return model

def preprocess_img(img,transform):
    img = transform(img)
    # add dimension for batch
    img = img.unsqueeze(0)
    return img


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    net = load_net("steerNet.pt")

    filenames = glob("../collect_data/data/*.jpg")

    for f in filenames:
        # read image from folder 
        image = cv2.imread(f)
        # image = cv2.resize(image, (84,84))
        #print(image.shape)
        # apply transorms
        input_img = preprocess_img(image,transform)
        #print(input_img.shape)
        steer = net(input_img).data.numpy()
        print(steer)




        
if __name__ == "__main__":
    main()