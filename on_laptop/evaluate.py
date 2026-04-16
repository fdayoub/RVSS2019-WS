#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from steer_net.steerDS import SteerDataSet
from steer_net.steerNet import SteerNet


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate steering model")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset = SteerDataSet(args.data_root, transform=transform, manifest_csv=args.manifest)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = SteerNet(mode="regression")
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()

    total = 0
    abs_sum = 0.0
    sq_sum = 0.0

    rows = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"]
            targets = batch["steering"].float().view(-1)
            preds = model(images).view(-1)
            errors = preds - targets

            abs_sum += torch.abs(errors).sum().item()
            sq_sum += torch.square(errors).sum().item()
            total += targets.numel()

            frame_ids = batch.get("frame_id")
            for i in range(targets.numel()):
                frame_id = frame_ids[i] if frame_ids is not None else ""
                rows.append(
                    [
                        frame_id,
                        float(targets[i].item()),
                        float(preds[i].item()),
                        float(errors[i].item()),
                    ]
                )

    mae = abs_sum / max(total, 1)
    rmse = math.sqrt(sq_sum / max(total, 1))

    metrics = {"samples": total, "mae": mae, "rmse": rmse}

    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_id", "true", "pred", "error"])
        writer.writerows(rows)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
