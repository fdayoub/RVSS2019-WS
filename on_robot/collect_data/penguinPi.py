import cv2
import numpy as np
import requests


def set_velocity(vel0, vel1, timeout=0.5):
    try:
        requests.get(
            "http://localhost:8080/robot/set/velocity?value=" + str(vel0) + "," + str(vel1),
            timeout=timeout,
        )
    except requests.RequestException:
        return False
    return True


def get_image(timeout=1.0):
    try:
        response = requests.get("http://localhost:8080/camera/get", timeout=timeout)
    except requests.RequestException:
        return None

    image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    return image
