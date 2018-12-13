#!/usr/bin/env python3
import time
import click
import math
import picamera
from picamera.array import PiRGBArray
import sys
sys.path.append("..")
import cv2
import numpy as np
import penguinPi as ppi
import pygame
import torch
from torch import nn
#~~~~~~~~~~~~ SET UP ROBOT ~~~~~~~~~~~~~~
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
#for teleop
pygame.init()
pygame.display.set_mode((10,10))
mA = ppi.Motor(ppi.AD_MOTOR_A)
mB = ppi.Motor(ppi.AD_MOTOR_B)
print("Initializing")
ppi.init()
mA.set_power(0)
mB.set_power(0)
# Teleop robot and save image taken as [leading count] + [steer angle between -0.5 and 0.5] .jpg 
try:
    angle = 0
    lead = 0
    while True:
        camera.capture(rawImage, format="bgr")
        image = rawImage.array
        image = cv2.resize(image, (84,84))
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    angle = 0
                if event.key == pygame.K_DOWN:
                    angle = 0
                if event.key == pygame.K_RIGHT:
                    angle += 0.05
                if event.key == pygame.K_LEFT:
                    angle -= 0.05
        angle = np.clip(angle, -0.5, 0.5)
        Kd = 15
        Ka = 25
        left  = int(Kd + Ka*angle)
        right = int(Kd - Ka*angle)
        mA.set_power(right)
        mB.set_power(left)
       
        cv2.imwrite("data/"+str(lead).zfill(6)+'%.2f'%angle+".jpg", image) 
        lead += 1
        rawImage.truncate(0)
except KeyboardInterrupt:
    mA.set_power(0)
    mB.set_power(0)

