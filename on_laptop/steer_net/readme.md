# In this folder

1. `steerDS`
   - A class to load and process images from a folder or a manifest CSV.
2. `steerNet`
   - A baseline network structure supporting regression (recommended) and classification outputs.

# Training workflow

1. Collect data on the robot and copy it into `../dev_data/`.
2. Use the root-level laptop training script:

```bash
python3 ../train.py \
  --data-root ../dev_data/session_x \
  --train-manifest ../dev_data/session_x/manifest.csv \
  --epochs 20 --batch-size 32 --lr 1e-3
```

3. Training outputs are saved to `on_laptop/runs/<run_name>/` with:
   - `best.pt`
   - `last.pt`
   - `history.csv`
   - `config.json`

# Manifest format

Expected headers include at least:
- `image_path`
- `steering`

Optional:
- `frame_id`
