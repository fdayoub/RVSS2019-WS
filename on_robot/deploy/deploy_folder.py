#!/usr/bin/env python3
import argparse
import csv
from glob import glob

import cv2
import torch
from torchvision import transforms

from steerNet import SteerNet


def parse_args():
    parser = argparse.ArgumentParser(description="Run model on image folder")
    parser.add_argument("--weights", default="steerNet.pt")
    parser.add_argument("--input-glob", default="../collect_data/data/*.jpg")
    parser.add_argument("--output-csv", default="folder_predictions.csv")
    parser.add_argument("--resize-width", type=int, default=84)
    parser.add_argument("--resize-height", type=int, default=84)
    return parser.parse_args()


def load_net(weights_filename):
    model = SteerNet(mode="regression")
    model.load_state_dict(torch.load(weights_filename, map_location="cpu"))
    model.eval()
    return model


def preprocess_img(img, transform, width, height):
    img = cv2.resize(img, (width, height))
    img = transform(img)
    return img.unsqueeze(0)


def main():
    args = parse_args()
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    net = load_net(args.weights)

    filenames = sorted(glob(args.input_glob))

    with open(args.output_csv, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["image_path", "steering_pred"])

        with torch.no_grad():
            for filename in filenames:
                image = cv2.imread(filename)
                if image is None:
                    print(f"warning: could not read {filename}")
                    continue

                input_img = preprocess_img(
                    image, transform, args.resize_width, args.resize_height
                )
                steer = float(net(input_img).item())
                writer.writerow([filename, f"{steer:.6f}"])
                print(filename, steer)


if __name__ == "__main__":
    main()
