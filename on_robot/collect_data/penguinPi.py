import numpy as np
import requests
import cv2      

def set_velocity(vel0, vel1):
    r = requests.get("http://localhost:8080/robot/set/velocity?value="+str(vel0)+","+str(vel1))

def get_image():
    r = requests.get("http://localhost:8080/camera/get")
    img = cv2.imdecode(np.fromstring(r.content,np.uint8), cv2.IMREAD_COLOR)

    return img
