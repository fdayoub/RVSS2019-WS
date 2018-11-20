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
# USE OWN NETWORK SIZE AND LOAD IN WEIGHTS
net = nn.Sequential(nn.Linear(84*29, 1280), nn.ReLU(), nn.Linear(1280, 512), nn.ReLU(), nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 1))
#net.load_state_dict(torch.load('net.pt'))
# Constants
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
        steer = net(torch.FloatTensor(strip).view(84*29)).data.numpy()
        steer = np.clip(steer, -0.5, 0.5)
        
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
