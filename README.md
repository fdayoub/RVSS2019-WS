# RVSS2019-WS
Repo for the workshop part of the [Australian Centre for Robotic Vision Summer School RVSS2019](https://www.roboticvision.org/rvss2019)

## The Workshop
Train a CNN and deploy it on a mobile robot to follow a track.
More [details can be found here](https://sites.google.com/view/rvss2019ws/overview).

## End-to-end quickstart

### 1) Collect data on robot

```bash
cd on_robot/collect_data
python3 collect.py --session-name session_01 --base-speed 50 --steer-step 0.1
```

This creates:
- `data/session_01/images/*.jpg`
- `data/session_01/manifest.csv`

### 2) Train on laptop

```bash
cd on_laptop
python3 train.py \
  --data-root ../on_robot/collect_data/data/session_01 \
  --train-manifest ../on_robot/collect_data/data/session_01/manifest.csv \
  --epochs 20 --batch-size 32 --lr 1e-3
```

Training outputs go to `on_laptop/runs/<run_name>/`.

### 3) Evaluate model offline

```bash
cd on_laptop
python3 evaluate.py \
  --weights runs/<run_name>/best.pt \
  --manifest ../on_robot/collect_data/data/session_01/manifest.csv \
  --data-root ../on_robot/collect_data/data/session_01 \
  --out-dir reports/<run_name>
```

Evaluation outputs:
- `metrics.json`
- `predictions.csv`

### 4) Deploy on robot

Copy `best.pt` to `on_robot/deploy/steerNet.pt`, then:

```bash
cd on_robot/deploy
python3 run_autonomy.py --weights steerNet.pt --base-speed 15 --steer-gain 25
```

## Project layout

- `on_laptop/`: dataset, model, training, evaluation
- `on_robot/collect_data/`: teleop data collection
- `on_robot/deploy/`: real-time inference + control
