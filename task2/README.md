# Task 2 — FNO for Dynamical System Identification

## Overview

This task trains Fourier Neural Operators (FNOs) to approximate the solution operator of an unknown 1D dynamical system from trajectory data. Rather than knowing the governing PDE, only snapshots of the system state at five time points are available. Four training strategies of increasing sophistication are explored: one-to-one endpoint mapping, resolution generalisation, all-to-all temporal modelling with FiLM conditioning, and transfer learning to an unknown initial-condition distribution.

---

## Problem

The unknown 1D PDE on `x ∈ [0, 1]`, `t ∈ (0, 1]` with zero boundary conditions:

```
∂_t u(x, t) = D(u)(x, t)
u(x, 0) = u₀(x),   u(0, t) = u(1, t) = 0
```

The operator D is unknown. The goal is to learn the solution operator `u₀ ↦ u(·, t)` from data alone.

### Dataset

Data files are **not included in the repository** — place them under `data/task2/` before running.

| File                                 | Shape          | Description                                |
|--------------------------------------|----------------|--------------------------------------------|
| `data_train_128.npy`                 | (1024, 5, 128) | Training trajectories                      |
| `data_val_128.npy`                   | (32,   5, 128) | Validation trajectories                    |
| `data_test_32.npy`                   | (128,  5,  32) | Test set at resolution 32                  |
| `data_test_64.npy`                   | (128,  5,  64) | Test set at resolution 64                  |
| `data_test_96.npy`                   | (128,  5,  96) | Test set at resolution 96                  |
| `data_test_128.npy`                  | (128,  5, 128) | Test set at resolution 128                 |
| `data_finetune_train_unknown_128.npy`| (32,   5, 128) | Fine-tune training (unknown distribution)  |
| `data_finetune_val_unknown_128.npy`  | (8,    5, 128) | Fine-tune validation (unknown distribution)|
| `data_test_unknown_128.npy`          | (128,  5, 128) | Test set (unknown distribution)            |

Each array has shape `(trajectories, time_snapshots, spatial_resolution)`. The five time snapshots correspond to `t ∈ {0.0, 0.25, 0.50, 0.75, 1.0}`.

---

## Files

| File                  | Purpose                                                       |
|-----------------------|---------------------------------------------------------------|
| `config.py`           | All hyperparameters, data paths, and model checkpoint paths   |
| `datasets.py`         | `One2OneDataset` and `All2AllDataset` PyTorch Dataset classes |
| `models/fno.py`       | `FNO1d` for one-to-one endpoint mapping                       |
| `models/fno_time.py`  | Time-conditioned `FNO1d` with FiLM layers                     |
| `utils.py`            | Training loop, relative L² error, checkpoint helpers          |
| `task1_one2one.py`    | Sub-task 1: train one-to-one FNO                              |
| `task2_resolution.py` | Sub-task 2: resolution-generalisation evaluation              |
| `task3_all2all.py`    | Sub-task 3: train all-to-all time-conditioned FNO             |
| `task4_finetune.py`   | Sub-task 4: fine-tune on unknown distribution                 |

---

## How to Run

All commands are run from the **project root**. Data must be placed under `data/task2/` first (see the Data section in the root README).

**Step 1 — One-to-one FNO**
```bash
python task2/task1_one2one.py
```

**Step 2 — Resolution generalisation (requires Step 1 checkpoint)**
```bash
python task2/task2_resolution.py
```

**Step 3 — All-to-all time-conditioned FNO**
```bash
python task2/task3_all2all.py
```

**Step 4 — Fine-tuning on unknown distribution (requires Step 3 checkpoint)**
```bash
python task2/task4_finetune.py
```

> To retrain from scratch: set `USE_PRETRAINED = False` in `task2/config.py`. Set it to `True` to load saved checkpoints and skip training.

---

## Approach

### FNO architecture
The core building block is a spectral convolution layer: the input is transformed to frequency space via FFT, the leading `modes = 16` Fourier modes are multiplied by learned complex weights, and the result is transformed back via IFFT. Three such spectral layers are stacked, each with a parallel 1D skip connection (`Conv1d`). Tanh activations are applied after each block.

### Two model variants

**`models/fno.py` — One-to-one FNO**
- Input channels: 2 (spatial coordinate x, initial condition u₀)
- Output: solution at t = 1.0 (1 channel)
- Parameters: 211,393
- Used in sub-tasks 1 and 2

**`models/fno_time.py` — Time-conditioned FNO with FiLM**
- Input channels: 3 (x, u(x, tᵢ), Δt)
- Output: solution at tⱼ (1 channel)
- Parameters: 212,609
- A FiLM (Feature-wise Linear Modulation) layer is inserted after each spectral block:
  ```
  FiLM(h, t) = h · (1 + γ(t)) + β(t)
  ```
  where γ and β are small MLPs conditioned on Δt. Both are zero-initialised so the network starts as an identity mapping and gradually learns temporal modulation.
- Used in sub-tasks 3 and 4

### All-to-all data augmentation
From each trajectory of 5 snapshots, all pairs (tᵢ, tⱼ) with tᵢ < tⱼ are generated at dataset construction time. This yields 10 pairs per trajectory, expanding 1,024 training trajectories to 10,240 training samples without collecting additional data.

### Optimiser
AdamW with a StepLR learning rate scheduler. One-to-one training runs for 100 epochs; all-to-all training runs for 250 epochs. Fine-tuning updates all model parameters (no frozen layers).

---

## Implementation Notes

- Spectral weights are initialised at scale `1 / (c_in · c_out)`, analogous to Xavier initialisation for the frequency domain.
- FiLM affine parameters γ and β are zero-initialised. This ensures the network outputs an identity transformation at epoch 0 and avoids gradient instability in early training.
- `All2AllDataset` pre-computes all valid (tᵢ, tⱼ) pairs at construction time and stores them as indexed tuples, so the training loop treats each pair as an independent sample.

---

## Results

### Sub-task 1 — One-to-one endpoint mapping

| Metric                          | Value  |
|---------------------------------|--------|
| Test relative L² error (t=1.0) | 1.71%  |
| Test trajectories               | 128    |
| Spatial resolution              | 128    |

### Sub-task 2 — Resolution generalisation (no retraining)

The one-to-one model is evaluated at spatial resolutions it was not trained on:

| Resolution | Relative L² Error |
|------------|-------------------|
| 32         | 24.00%            |
| 64         |  7.53%            |
| 96         |  4.19%            |
| 128        |  1.71%            |

The large error at resolution 32 is expected: resolution 32 is the Nyquist limit for 16 Fourier modes (2 × 16 = 32), so the spectral representation is at the aliasing boundary and cannot represent the input faithfully.

### Sub-task 3 — All-to-all time-conditioned FNO

All predictions are made directly from u₀ (not autoregressively), so errors do not compound across time steps.

| Time step | Relative L² Error |
|-----------|-------------------|
| t = 0.25  | 0.72%             |
| t = 0.50  | 0.70%             |
| t = 0.75  | 0.69%             |
| t = 1.00  | 0.72%             |

The all-to-all model achieves a **2.4× improvement** over the one-to-one baseline at t = 1.0 (0.72% vs. 1.71%). Errors are stable across time steps because all predictions are conditioned independently on u₀.

### Sub-task 4 — Transfer learning on unknown distribution (at t = 1.0)

| Approach       | Training data       | Relative L² Error |
|----------------|---------------------|-------------------|
| Zero-shot      | 0 trajectories      | 3.29%             |
| Fine-tuned     | 32 trajectories     | 3.07%             |
| From scratch   | 32 trajectories     | 9.93%             |

The fine-tuned model is **3.2× better** than training from scratch with the same 32 trajectories, confirming that the pretrained operator has learned reusable structure. The marginal improvement from zero-shot to fine-tuned (3.29% → 3.07%) also suggests the pretrained model generalises substantially even without adaptation.

---

## Plots

Numerical results for Task 2 are reported in the tables above and in `report/report.pdf`. Training curves and prediction visualisations can be added by extending `utils.py` with a plotting helper analogous to the one in `task1/utils_task1.py`.
