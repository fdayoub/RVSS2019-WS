# In this folder

1. Copy your network and its weights here (see `steerNet.py` and `steerNet.pt` examples).
2. Run autonomous steering using:

```bash
python3 run_autonomy.py --weights steerNet.pt --base-speed 15 --steer-gain 25
```

3. For offline debugging on saved images:

```bash
python3 deploy_folder.py --input-glob "../collect_data/data/*.jpg" --output-csv folder_predictions.csv
```

## Troubleshooting

- Camera unavailable: verify robot services on `localhost:8080` are running.
- Model shape mismatch: ensure training/deploy model architecture modes match.
- Unstable steering: lower `--steer-gain` and/or `--steer-clamp`.
