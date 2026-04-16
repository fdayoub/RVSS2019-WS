#!/usr/bin/env python3
import argparse
import time

import cv2
import numpy as np
import torch
from torchvision import transforms

import penguinPi as ppi
from steerNet import SteerNet


def parse_args():
    parser = argparse.ArgumentParser(description="Run autonomous steering on robot")
    parser.add_argument("--weights", default="steerNet.pt")
    parser.add_argument("--base-speed", type=int, default=15)
    parser.add_argument("--steer-gain", type=float, default=25.0)
    parser.add_argument("--camera-width", type=int, default=84)
    parser.add_argument("--camera-height", type=int, default=84)
    parser.add_argument("--max-runtime-sec", type=float, default=0.0)
    parser.add_argument("--steer-clamp", type=float, default=0.7)
    parser.add_argument("--ema-alpha", type=float, default=0.3)
    return parser.parse_args()


def load_model(weights_filename):
    model = SteerNet(mode="regression")
    model.load_state_dict(torch.load(weights_filename, map_location="cpu"))
    model.eval()
    return model


def preprocess_image(image, transform, width, height):
    image = cv2.resize(image, (width, height))
    tensor = transform(image)
    return tensor.unsqueeze(0)


def main():
    args = parse_args()
    model = load_model(args.weights)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    ppi.set_velocity(0, 0)
    started = time.time()
    steering_filtered = 0.0
    frames = 0

    try:
        with torch.no_grad():
            while True:
                if args.max_runtime_sec > 0 and (time.time() - started) >= args.max_runtime_sec:
                    print("Reached max runtime; stopping.")
                    break

                image = ppi.get_image()
                if image is None:
                    print("warning: camera frame unavailable")
                    ppi.set_velocity(0, 0)
                    time.sleep(0.05)
                    continue

                input_img = preprocess_image(image, transform, args.camera_width, args.camera_height)
                steer = float(model(input_img).item())
                steer = float(np.clip(steer, -args.steer_clamp, args.steer_clamp))
                steering_filtered = (args.ema_alpha * steer) + ((1.0 - args.ema_alpha) * steering_filtered)

                left = int(args.base_speed + args.steer_gain * steering_filtered)
                right = int(args.base_speed - args.steer_gain * steering_filtered)
                ppi.set_velocity(left, right)

                frames += 1
                if frames % 10 == 0:
                    elapsed = max(time.time() - started, 1e-6)
                    fps = frames / elapsed
                    print(
                        f"steer={steer:.3f} filtered={steering_filtered:.3f} "
                        f"left={left} right={right} fps={fps:.1f}"
                    )

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        ppi.set_velocity(0, 0)
        print("Robot stopped")


if __name__ == "__main__":
    main()
