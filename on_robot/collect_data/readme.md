# In this folder
Use/modify `collect.py` to collect images and steering data while driving the robot over the track.

## Quick start

```bash
python3 collect.py --session-name session_trial_01 --base-speed 50 --steer-step 0.1
```

Data is saved under `data/<session_name>/` with:
- `images/<frame>.jpg`
- `manifest.csv` with frame path, steering, motor velocities, and timestamp.

## Recommended collection protocol

1. Capture at least 3 sessions on-track.
2. Include clockwise and counter-clockwise laps.
3. Include some variation in lighting/backgrounds.
4. Aim for balanced left/right/straight steering examples.

## Data quality checklist

- Robot moves continuously; avoid long stationary segments.
- Camera feed is not overexposed/underexposed.
- Steering is varied enough to include turns and straight segments.
- Press `SPACE` at any time for emergency stop.
