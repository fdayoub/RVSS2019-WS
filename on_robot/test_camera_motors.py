#!/usr/bin/env python3
import deploy.penguinPi as ppi
import time

if __name__ == "__main__":
    print("test left motor")
    ppi.set_velocity(10,0)
    time.sleep(3)
    print("test right motor")
    ppi.set_velocity(0,10)
    time.sleep(3)
    print("stop")
    ppi.set_velocity(0,0)
    print("capture image")
    image = ppi.get_image()
    print("image size %d by %d" % (image.shape[0],image.shape[1]))

