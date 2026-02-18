# Task 2: Training a Fourier Neural Operator for Dynamical System Identification

Trains 1D Fourier Neural Operators (FNOs) to learn the solution operator of an unknown dynamical system from trajectory data. The task progresses from a simple one-to-one mapping, through resolution generalisation, to a time-conditional all-to-all model, and finally fine-tuning on an unknown distribution.

---

## Directory Structure

```
task2/
├── config.py              # Central configuration (paths, hyperparameters, flags)
├── datasets.py            # PyTorch Dataset classes (One2One, All2All)
├── models/
│   ├── fno.py             # FNO1d — standard one-to-one model
│   └── fno_time.py        # FNO1d with FILM time-conditioning (all-to-all)
├── utils.py               # Training loop and evaluation utilities
├── task1_one2one.py       # Sub-task 1 — one-to-one FNO (10 pts)
├── task2_resolution.py    # Sub-task 2 — resolution generalisation test (10 pts)
├── task3_all2all.py       # Sub-task 3 — all-to-all time-conditional FNO (15 pts)
├── task4_finetune.py      # Sub-task 4 — fine-tuning on unknown distribution (15 + 10 pts)
└── description.md         # Official course task description
```

Outputs are written to (relative to project root):

```
models/                    # Checkpoints: task1_one2one.pt, task3_all2all.pt,
                           #              task4_finetuned.pt, task4_scratch.pt
plots/task2/               # Training curves and evaluation figures
```

---

## Data

Place the `.npy` trajectory files under `data/task2/` (see the root [README](../README.md#task-2) for the expected filenames and shapes). The paths are configured in [config.py](config.py) under `DATA_PATHS`.

---

## Configuration

All hyperparameters and data paths are in [config.py](config.py). The two flags most likely to need changing are:

| Key | Default | Description |
|---|---|---|
| `DEVICE` | `"cpu"` | Set to `"cuda"` to use GPU |
| `USE_PRETRAINED` | `True` | Load saved checkpoints instead of retraining |

Data paths (`DATA_PATHS`) and model checkpoint paths (`MODEL_PATHS`) can also be updated in `config.py` if your files live elsewhere.

---

## Running

All scripts are run from the **project root** with the virtual environment active.

### Sub-task 1 — One-to-one FNO

```bash
python task2/task1_one2one.py
```

Trains an FNO1d to map the initial condition `u(x, t=0)` directly to the final state `u(x, t=1.0)`. With `USE_PRETRAINED = True` loads `models/task1_one2one.pt` and skips training. Prints relative L² error on 128 test trajectories.

### Sub-task 2 — Resolution generalisation

```bash
python task2/task2_resolution.py
```

Loads the sub-task 1 checkpoint and evaluates it at four spatial resolutions (32, 64, 96, 128). Prints a resolution vs. error table — no training is performed.

### Sub-task 3 — All-to-all time-conditional FNO

```bash
python task2/task3_all2all.py
```

Trains an FNO1d with a FILM conditioning layer to map any `(u(x, tᵢ), Δt)` pair to `u(x, tⱼ)` for all `i < j` across the five time snapshots. Saves checkpoint to `models/task3_all2all.pt`. Reports errors both at `t=1.0` (for comparison with sub-task 1) and at intermediate steps.

### Sub-task 4 — Fine-tuning on unknown distribution

```bash
python task2/task4_finetune.py
```

Runs three experiments in sequence:

1. **Zero-shot** — evaluates the sub-task 3 model on `data_test_unknown_128.npy` with no adaptation.
2. **Fine-tuning** — continues training from the sub-task 3 checkpoint on 32 unknown-distribution trajectories; saves to `models/task4_finetuned.pt`.
3. **Training from scratch (bonus)** — trains a fresh model on the same 32 samples for comparison; saves to `models/task4_scratch.pt`.

---

## Model Architecture

**FNO1d** (`models/fno.py`) — spectral convolution layers that lift the input to a hidden channel width, apply Fourier-domain filters for `modes` frequencies, and project back to the output.

**FNO1d + FILM** (`models/fno_time.py`) — same architecture extended with Feature-wise Linear Modulation: a small MLP maps the scalar `Δt` to per-channel scale and bias that modulate each spectral layer, enabling time-conditioned predictions.

Default hyperparameters: `modes = 16`, `width = 64`.
