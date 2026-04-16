#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from steer_net.steerDS import SteerDataSet
from steer_net.steerNet import SteerNet


def parse_args():
    parser = argparse.ArgumentParser(description="Train steering model")
    parser.add_argument("--data-root", required=True, help="Root directory for images")
    parser.add_argument("--train-manifest", default=None, help="CSV manifest for train data")
    parser.add_argument("--val-manifest", default=None, help="CSV manifest for val data")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split when val manifest absent")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_datasets(args, transform):
    if args.train_manifest and args.val_manifest:
        train_ds = SteerDataSet(args.data_root, transform=transform, manifest_csv=args.train_manifest)
        val_ds = SteerDataSet(args.data_root, transform=transform, manifest_csv=args.val_manifest)
        return train_ds, val_ds

    if not args.train_manifest:
        raise ValueError("--train-manifest is required unless both train/val manifests are provided")

    full_ds = SteerDataSet(args.data_root, transform=transform, manifest_csv=args.train_manifest)
    val_size = int(len(full_ds) * args.val_split)
    train_size = len(full_ds) - val_size
    generator = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(full_ds, [train_size, val_size], generator=generator)
    return train_ds, val_ds


def evaluate_loss(model, loader, criterion):
    model.eval()
    running = 0.0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"]
            targets = batch["steering"].float().view(-1, 1)
            preds = model(images)
            loss = criterion(preds, targets)
            running += loss.item() * images.size(0)
    return running / max(len(loader.dataset), 1)


def train():
    args = parse_args()
    set_seed(args.seed)

    run_name = args.run_name or datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_ds, val_ds = make_datasets(args, transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = SteerNet(mode="regression")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.SmoothL1Loss()

    best_val = float("inf")
    history_path = os.path.join(run_dir, "history.csv")
    with open(history_path, "w", newline="", encoding="utf-8") as history_file:
        writer = csv.writer(history_file)
        writer.writerow(["epoch", "train_loss", "val_loss"])

        for epoch in range(1, args.epochs + 1):
            model.train()
            running = 0.0
            for batch in train_loader:
                images = batch["image"]
                targets = batch["steering"].float().view(-1, 1)

                optimizer.zero_grad()
                preds = model(images)
                loss = criterion(preds, targets)
                loss.backward()
                optimizer.step()

                running += loss.item() * images.size(0)

            train_loss = running / max(len(train_loader.dataset), 1)
            val_loss = evaluate_loss(model, val_loader, criterion)
            writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}"])
            history_file.flush()

            torch.save(model.state_dict(), os.path.join(run_dir, "last.pt"))
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), os.path.join(run_dir, "best.pt"))

            print(f"epoch {epoch}/{args.epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Training complete. Artifacts written to: {run_dir}")


if __name__ == "__main__":
    train()
