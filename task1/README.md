# Task 1: Visualizing Loss Landscapes — PINNs vs. Data-Driven Solvers

Compares Physics-Informed Neural Networks (PINNs) and supervised Data-Driven (DD) solvers on the 2D Poisson equation across varying frequency complexities (K = 1, 4, 16), and visualises the loss landscape geometry around each trained model.

---

## Directory Structure

```
task1/
├── config_task1.py      # Central configuration (hyperparameters, paths, flags)
├── physics.py           # Poisson equation physics: source term and analytical solution
├── models.py            # MLP, PoissonPINN, and DataDrivenSolver classes
├── utils_task1.py       # Plotting and evaluation helpers
├── task1_1_data.py      # Sub-task 1.1 — data generation and visualisation
├── task1_2_train.py     # Sub-task 1.2 — training PINN and DD solvers
├── task1_3_landscape.py # Sub-task 1.3 (bonus) — loss landscape visualisation
└── description.md       # Official course task description
```

Outputs are written to (relative to project root):

```
plots/task1_1/           # Source / solution sample plots
plots/task1_2/           # Training curves and prediction comparisons
plots/task1_3/           # Contour and 3-D surface plots of loss landscapes
plots/task1_3/cache/     # HDF5 landscape cache (git-ignored)
models/                  # Checkpoints: task1_pinn_K{k}.pt, task1_dd_K{k}.pt
```

---

## Configuration

All hyperparameters are in [config_task1.py](config_task1.py). The two flags most likely to need changing are:

| Key | Default | Description |
|---|---|---|
| `DEVICE` | `"cpu"` | Set to `"cuda"` to use GPU |
| `USE_PRETRAINED` | `True` | Load saved checkpoints instead of retraining |

Other notable settings: grid resolution `N = 64`, frequency levels `K_values = [1, 4, 16]`, and the landscape cache mode `cache_mode = "use"` (set to `"recompute"` to force recomputation).

---

## Running

All scripts are run from the **project root** with the virtual environment active.

### Sub-task 1.1 — Generate and visualise data

```bash
python task1/task1_1_data.py
```

Generates the source term `f` and analytical solution `u` on a 64×64 grid for each K value. Saves sample plots to `plots/task1_1/`.

### Sub-task 1.2 — Train PINN and Data-Driven solvers

```bash
python task1/task1_2_train.py
```

Trains both a PINN and a DD solver for each K. With `USE_PRETRAINED = True` it skips training and loads existing checkpoints. PINN uses a two-phase schedule (Adam → L-BFGS); DD uses Adam only. Outputs checkpoints to `models/`, comparison plots to `plots/task1_2/`, and prints relative L² errors.

### Sub-task 1.3 — Visualise loss landscapes (bonus)

```bash
python task1/task1_3_landscape.py
```

Loads the trained models and computes the 2-D loss landscape around θ* via two random filter-normalised directions. Outputs contour and 3-D surface plots to `plots/task1_3/`. Results are cached in HDF5 under `plots/task1_3/cache/` so subsequent runs are fast.
