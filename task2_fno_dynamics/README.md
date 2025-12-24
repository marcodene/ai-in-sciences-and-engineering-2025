# Task 2: FNO Dynamics - Complete Implementation Guide

## Overview

This directory contains the complete implementation for Task 2 of the AISE project, focusing on training Fourier Neural Operators (FNO) to approximate dynamical systems.

## Project Structure

```
task2_fno_dynamics/
├── models/
│   ├── fno.py              # Basic FNO (Task 1: one2one)
│   └── fno_time.py         # Time-conditional FNO with FILM (Task 3: all2all)
├── train_one2one.py        # Task 1 & 2: One-to-one training
├── train_all2all.py        # Task 3: All-to-all training
├── train_finetune.py       # Task 4: Finetuning on unknown distribution
├── visualize.py            # Visualization utilities
├── description.md          # Full task description
└── DIMENSIONI_GUIDA.md     # Comprehensive dimension guide (Italian)
```

## Tasks Implementation

### Task 1: One-to-One Training (10 points) ✅

**Objective**: Train FNO to learn mapping `u(t=0) → u(t=1.0)`

**Script**: `train_one2one.py`

```bash
python task2_fno_dynamics/train_one2one.py
```

**What it does**:
- Trains a basic FNO on 1024 trajectories
- Uses only initial condition (t=0) and final state (t=1.0)
- Saves model to `models/task2_one2one.pt`
- Reports average relative L² error on test set

---

### Task 2: Multi-Resolution Testing (10 points) ✅

**Objective**: Test Task 1 model on different spatial resolutions

**Script**: Same as Task 1 - `train_one2one.py` (runs automatically)

**What it does**:
- Tests the one2one model on resolutions: 32, 64, 96, 128
- Demonstrates FNO's resolution independence
- Reports errors for each resolution

---

### Task 3: All-to-All Training (15 points) ✅

**Objective**: Train time-dependent FNO using all time snapshots

**Script**: `train_all2all.py`

```bash
python task2_fno_dynamics/train_all2all.py
```

**What it does**:
- **Part 1 (10 points)**: Tests at t=1.0 and compares with Task 1
- **Part 2 (5 points)**: Tests at multiple timesteps (0.25, 0.50, 0.75, 1.0)
- Uses time-conditional FNO with FILM layers
- Trains on ALL forward time pairs (10 pairs per trajectory)
- Uses **delta time (Δt)** approach (aligned with professor's method)
- Saves model to `models/task2_all2all.pt`

**Key Features**:
- Input: `[x-coordinates, u(x, t_i), Δt]`
- Output: `u(x, t_j)` where `Δt = t_j - t_i`
- FILM layers for time conditioning

---

### Task 4: Finetuning (25 points total) ✅

**Objective**: Test and adapt model on unknown initial condition distribution

**Script**: `train_finetune.py`

```bash
python task2_fno_dynamics/train_finetune.py
```

**What it does**:

1. **Task 4.1 - Zero-Shot Testing (5 points)**:
   - Tests pretrained Task 3 model on unknown distribution
   - No adaptation, just evaluation
   
2. **Task 4.2 - Finetuning (10 points)**:
   - Finetunes Task 3 model using 32 trajectories from unknown distribution
   - Uses lower learning rate and fewer epochs
   - Saves to `models/task2_finetuned.pt`
   
3. **Task 4.3 - Train from Scratch (10 bonus points)**:
   - Trains new model from scratch using only 32 trajectories
   - Compares with finetuned model to assess transfer learning
   - Saves to `models/task2_scratch.pt`

**Final Output**: Comparison showing if transfer learning is successful

---

## Running All Tasks

To complete the entire assignment, run in order:

```bash
# Task 1 & 2
python task2_fno_dynamics/train_one2one.py

# Task 3
python task2_fno_dynamics/train_all2all.py

# Task 4
python task2_fno_dynamics/train_finetune.py
```

## Key Implementation Details

### Delta Time Approach

Following the professor's tutorial, we use **delta time (Δt)** instead of absolute time:

- **Input**: `[x, u(t_i), Δt]` where `Δt = t_j - t_i`
- **Advantage**: No ambiguity - model learns to evolve state by Δt
- **Training pairs**: All forward transitions (t_i → t_j where i < j)
  - (0→1), (0→2), (0→3), (0→4)
  - (1→2), (1→3), (1→4)
  - (2→3), (2→4)
  - (3→4)
  - **Total**: 10 pairs per trajectory

### FILM Time Conditioning

The time-conditional FNO uses Feature-wise Linear Modulation (FILM):

```python
x_modulated = x * (1 + scale(time)) + bias(time)
```

This allows the network to adapt its features based on the time evolution parameter.

### Error Metric

All tasks use **average relative L² error**:

```
error = (1/N) * Σ ||u_pred - u_true||₂ / ||u_true||₂
```

## Hyperparameters

### Task 1 & 2 (one2one):
- Modes: 16
- Width: 64
- Batch size: 32
- Learning rate: 0.001
- Epochs: 100

### Task 3 (all2all):
- Modes: 16
- Width: 64
- Batch size: 32
- Learning rate: 0.001
- Epochs: 30

### Task 4 (finetuning):
- Finetuning LR: 0.0005 (lower!)
- Finetuning epochs: 50
- From-scratch LR: 0.001
- From-scratch epochs: 200 (more needed!)

## Expected Results

Based on typical FNO performance:

- **Task 1**: Error ~0.01-0.03 at t=1.0
- **Task 2**: Errors increase slightly with lower resolutions
- **Task 3**: Error ~0.005-0.02 at t=1.0 (better than Task 1)
- **Task 4**: Finetuning should significantly reduce zero-shot error

## Troubleshooting

### "Pretrained model not found"
Make sure to run `train_all2all.py` before `train_finetune.py`

### Out of memory
Reduce batch size in the scripts

### Poor convergence
- Increase number of epochs
- Adjust learning rate
- Check data normalization

## References

- Original FNO Paper: Li et al. (2021) - Fourier Neural Operator for Parametric PDEs
- FILM conditioning: Perez et al. (2018) - FiLM: Visual Reasoning with a General Conditioning Layer

## Author Notes

This implementation uses **delta time (Δt)** following the professor's approach in the CNO tutorial, which:
- Avoids temporal ambiguity
- Works well for autonomous PDEs
- Provides good generalization across different time intervals
