#!/usr/bin/env python3
import argparse
import sys
import time

import deploy.penguinPi as ppi


def parse_args():
    parser = argparse.ArgumentParser(description="Basic camera and motor smoke test")
    parser.add_argument("--left-vel", type=int, default=10)
    parser.add_argument("--right-vel", type=int, default=10)
    parser.add_argument("--duration-sec", type=float, default=3.0)
    return parser.parse_args()


def main():
    args = parse_args()

    print("test left motor")
    ppi.set_velocity(args.left_vel, 0)
    time.sleep(args.duration_sec)

    print("test right motor")
    ppi.set_velocity(0, args.right_vel)
    time.sleep(args.duration_sec)

    print("stop")
    ppi.set_velocity(0, 0)

    print("capture image")
    image = ppi.get_image()
    if image is None:
        print("camera capture failed")
        return 1

    print("image size %d by %d" % (image.shape[0], image.shape[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
