# Pacman Behavior Cloning

A Pac-Man game with an AI agent built on XGBoost.

## Project Structure

```
project/
├── run.py                        # Main entry point
├── requirements.txt              # Game dependencies (pygame, Pillow)
├── assignment/
│   ├── recorder.py               # Records gameplay frames to CSV
│   ├── autoplay.py               # Loads the trained model and controls Pacman
│   ├── train.py                  # Trains an XGBoost classifier on recorded data
│   ├── requirements.txt          # ML dependencies
│   ├── model/                    # Saved model & label encoder (created after training)
│   ├── checkpoints/              # XGBoost checkpoints saved during training
│   └── scripts/
│       ├── helper.py             # Migration script v1 → adds is_reversing_* columns
│       └── helper2.py            # Migration script v1/v2 → unified moving_dir format
├── recordings/                   # CSV files recorded during gameplay
├── assets/                       # Fonts, images, sounds, maps
└── pacman/                       # Game engine source code
```


## Setup

**1. Install game dependencies:**
```bash
pip install -r requirements.txt
```

**2. Install ML dependencies:**
```bash
pip install -r assignment/requirements.txt
```


## Usage

### Step 1 — Record your gameplay

Run the game with recording enabled. Every frame of your session is saved as a row in a CSV file under `recordings/`.

```bash
python run.py --record
```

Optional: specify a custom output directory:
```bash
python run.py --record --record-dir my_recordings
```

### Step 2 — Train the model

Train an XGBoost classifier on your recorded sessions.

```bash
python assignment/train.py
```

Key options:

| Flag                    | Default      | Description                                     |
| ----------------------- | ------------ | ----------------------------------------------- |
| `--data-dir`            | `recordings` | Directory containing recorded CSV files         |
| `--model-dir`           | `assignment` | Output directory; model saved to `<dir>/model/` |
| `--checkpoint-interval` | `100`        | Save a checkpoint every N boosting rounds       |
| `--test-size`           | `0.2`        | Fraction of episodes held out for evaluation    |
| `--seed`                | `0`          | Random seed for reproducibility                 |
| `--eval-only`           | —            | Skip training; evaluate an existing model       |

After training, the model and label encoder are saved to `assignment/model/`.


### Autoplay

Load the trained model and let it control Pacman autonomously:

```bash
python run.py --autoplay
```

Optional: point to a different model directory:
```bash
python run.py --autoplay --model-dir assignment
```
