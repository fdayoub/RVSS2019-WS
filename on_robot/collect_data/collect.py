#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import os
import time

import cv2
import numpy as np
import pygame

import penguinPi as ppi


def parse_args():
    parser = argparse.ArgumentParser(description="Collect steering data while teleoperating robot")
    parser.add_argument("--out-dir", default="data", help="Base output directory for sessions")
    parser.add_argument(
        "--session-name",
        default=dt.datetime.utcnow().strftime("session_%Y%m%d_%H%M%S"),
        help="Session folder name (default: timestamp)",
    )
    parser.add_argument("--base-speed", type=int, default=50, help="Base motor speed")
    parser.add_argument("--steer-step", type=float, default=0.1, help="Steering increment per key press")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames to save (0 disables limit)")
    return parser.parse_args()


def setup_output(out_dir, session_name):
    session_dir = os.path.join(out_dir, session_name)
    images_dir = os.path.join(session_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    manifest_path = os.path.join(session_dir, "manifest.csv")

    csv_file = open(manifest_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(
        [
            "frame_id",
            "image_path",
            "steering",
            "base_speed",
            "left_vel",
            "right_vel",
            "timestamp_ms",
        ]
    )
    return session_dir, images_dir, manifest_path, csv_file, writer


def main():
    args = parse_args()

    pygame.init()
    pygame.display.set_mode((100, 100))

    session_dir, images_dir, manifest_path, csv_file, writer = setup_output(args.out_dir, args.session_name)

    frame_id = 0
    angle = 0.0
    started_ms = int(time.time() * 1000)

    ppi.set_velocity(0, 0)
    print(f"Saving data to: {session_dir}")

    try:
        while True:
            image = ppi.get_image()
            if image is None:
                print("warning: camera image unavailable, skipping frame")
                time.sleep(0.05)
                continue

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        angle = 0.0
                        print("straight")
                    if event.key == pygame.K_DOWN:
                        angle = 0.0
                        print("straight")
                    if event.key == pygame.K_RIGHT:
                        angle += args.steer_step
                        print(f"right (angle={angle:.2f})")
                    if event.key == pygame.K_LEFT:
                        angle -= args.steer_step
                        print(f"left (angle={angle:.2f})")
                    if event.key == pygame.K_SPACE:
                        print("stop")
                        raise KeyboardInterrupt

            angle = float(np.clip(angle, -0.5, 0.5))
            left_vel = int(args.base_speed + args.base_speed * angle)
            right_vel = int(args.base_speed - args.base_speed * angle)

            ppi.set_velocity(left_vel, right_vel)

            filename = f"{frame_id:06d}.jpg"
            image_rel_path = os.path.join("images", filename)
            image_abs_path = os.path.join(session_dir, image_rel_path)
            saved = cv2.imwrite(image_abs_path, image)
            if not saved:
                print(f"warning: failed to write frame {frame_id}")
                continue

            timestamp_ms = int(time.time() * 1000)
            writer.writerow(
                [
                    frame_id,
                    image_rel_path,
                    f"{angle:.4f}",
                    args.base_speed,
                    left_vel,
                    right_vel,
                    timestamp_ms,
                ]
            )
            csv_file.flush()

            frame_id += 1
            if args.max_frames > 0 and frame_id >= args.max_frames:
                print(f"Reached --max-frames={args.max_frames}; stopping.")
                break

    except KeyboardInterrupt:
        pass
    finally:
        ppi.set_velocity(0, 0)
        csv_file.close()
        elapsed_sec = (int(time.time() * 1000) - started_ms) / 1000.0
        print("Collection complete")
        print(f"  session: {session_dir}")
        print(f"  manifest: {manifest_path}")
        print(f"  frames: {frame_id}")
        print(f"  duration_sec: {elapsed_sec:.2f}")


if __name__ == "__main__":
    main()
