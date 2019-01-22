#!/usr/bin/env python3
import time
import click
import math
import picamera
from picamera.array import PiRGBArray
import cv2
import numpy as np
import penguinPi as ppi
import pygame
import torch
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


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
net = load_net("steerNet.pt")

IM_WIDTH = 160
IM_HEIGHT = 120
minimum_area = 25
# Camera Setup
camera = picamera.PiCamera()
camera.rotation = 180
camera.resolution = (IM_WIDTH, IM_HEIGHT)
# Image holder
rawImage = PiRGBArray(camera, size=(IM_WIDTH, IM_HEIGHT))
time.sleep(1)
#~~~~~~~~~~~~ SET UP ROBOT ~~~~~~~~~~~~~~
mA = ppi.Motor("AD_MOTOR_R")
mB = ppi.Motor("AD_MOTOR_L")
print("Initializing")
ppi.init()
mA.set_velocity(0)
mB.set_velocity(0)
try:
    angle = 0
    while True:
        camera.capture(rawImage, format="bgr")
        image = rawImage.array
        image = cv2.resize(image, (84,84))
        # EDIT INPUT SIZE
        input_img = preprocess_img(image,transform)
        direc = net(input_img).data.numpy()
        direc = np.argmax(direc)
        if direc == 0:
            steer = 0.1
        elif direc == 1:
            steer = -0.1
        else:
            steer = 0
       
        print(steer)
        Kd = 15 
        Ka = 25
        left  = int(Kd + Ka*steer)
        right = int(Kd - Ka*steer)
        
        mA.set_velocity(right)
        mB.set_velocity(left)
        
        rawImage.truncate(0)
except KeyboardInterrupt:
    mA.set_velocity(0)
    mB.set_velocity(0)
