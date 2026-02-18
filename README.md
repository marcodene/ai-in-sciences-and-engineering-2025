# AISE 2026 Project: Neural PDE Solvers

A course project for **AI in the Sciences and Engineering (HS25)** at ETH Zurich, by **Marco De Negri**. The project explores three complementary approaches to learning solutions of partial differential equations with neural networks: (1) comparing loss landscapes of Physics-Informed Neural Networks (PINNs) against supervised data-driven solvers on the 2D Poisson equation, (2) training Fourier Neural Operators (FNOs) to identify an unknown dynamical system from trajectory data, and (3) extending the Geometry-Aware Operator Transformer (GAOT) to support irregular-geometry tokenization via random sampling and a dynamic radius strategy.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Prerequisites & Installation](#prerequisites--installation)
3. [Data](#data)
4. [Running the Project](#running-the-project)
   - [Task 1 — Loss Landscapes](#task-1--loss-landscapes-pinns-vs-data-driven)
   - [Task 2 — FNO Dynamical System](#task-2--fno-dynamical-system-identification)
   - [Task 3 — GAOT Extension](#task-3--gaot-extension-for-irregular-geometries)
5. [Results](#results)
6. [Report](#report)
7. [License](#license)

---

## Project Structure

```
aise2026_project/
│
├── task1/                          # Task 1: Loss landscapes (PINNs vs. Data-Driven)
│   ├── config_task1.py             # Hyperparameters and paths for Task 1
│   ├── description.md              # Official task description
│   ├── models.py                   # MLP, PoissonPINN, and DataDrivenSolver classes
│   ├── physics.py                  # Analytical Poisson source term and solution
│   ├── utils_task1.py              # Plotting and utility functions
│   ├── task1_1_data.py             # Task 1.1: data generation and visualisation
│   ├── task1_2_train.py            # Task 1.2: train PINN and Data-Driven models
│   └── task1_3_landscape.py        # Task 1.3: loss landscape visualisation (bonus)
│
├── task2/                          # Task 2: FNO for dynamical system identification
│   ├── config.py                   # Hyperparameters, data paths, and model paths
│   ├── description.md              # Official task description
│   ├── datasets.py                 # PyTorch Dataset classes (One2One, All2All)
│   ├── utils.py                    # Training loop, metrics, and checkpoint helpers
│   ├── models/
│   │   ├── fno.py                  # FNO1d for one-to-one mapping
│   │   └── fno_time.py             # Time-conditioned FNO1d with FiLM layers
│   ├── task1_one2one.py            # Task 2.1: one-to-one FNO training
│   ├── task2_resolution.py         # Task 2.2: resolution-generalisation test
│   ├── task3_all2all.py            # Task 2.3: all-to-all time-dependent FNO
│   └── task4_finetune.py           # Task 2.4: fine-tuning on unknown distribution
│
├── task3/                          # Task 3: GAOT extension for irregular geometries
│   ├── description.md              # Official task description
│   ├── GAOT-base/                  # Baseline GAOT implementation (Strategy I)
│   │   ├── main.py                 # Entry point
│   │   ├── README.md               # GAOT-base usage instructions
│   │   ├── requirements.txt        # Task-3-specific dependencies
│   │   ├── config/examples/        # Example JSON/TOML config files
│   │   ├── datasets/               # Datasets folder (NetCDF .nc files go here)
│   │   └── src/                    # Core source: model, trainer, datasets, utils
│   └── GAOT-random-sampling-dynamic-radius/   # Extended GAOT (Strategy II)
│       ├── main.py                 # Entry point
│       ├── visualize_tokenization.py  # Visualise random-sampling token strategies
│       ├── README.md               # Extended GAOT usage instructions
│       ├── requirements.txt        # Task-3-specific dependencies (same as base)
│       ├── config/examples/        # Example config files for the extended model
│       ├── datasets/               # Datasets folder
│       ├── plots/                  # Tokenization strategy visualisations
│       └── src/                    # Extended source with dynamic-radius graph builder
│
├── models/                         # Trained PyTorch checkpoints (git-ignored, *.pt)
├── plots/                          # Generated figures organised by sub-task
│   ├── task1_1/                    # Source/solution sample plots
│   ├── task1_2/                    # Training curves and prediction comparisons
│   └── task1_3/                    # Loss landscape contour and surface plots
│       └── cache/                  # HDF5 landscape cache (git-ignored)
│
├── report/
│   ├── report.tex                  # LaTeX source of the final report
│   ├── references.bib              # Bibliography
│   └── report.pdf                  # Compiled final report (deliverable)
│
├── data/                           # Datasets — git-ignored; see Data section below
├── venv/                           # Python virtual environment — git-ignored
├── requirements.txt                # Root dependencies for Task 1 and Task 2
├── .gitignore
└── aise2026-project-description.pdf  # Official course project PDF
```

---

## Prerequisites & Installation

**Python version:** 3.13 (used during development; 3.10+ should also work)

### 1. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows
```

### 2. Install root dependencies (Task 1 & Task 2)

```bash
pip install -r requirements.txt
```

### 3. Install Task 3 dependencies

Task 3 uses a larger set of packages (including `torch-geometric`, `torch-scatter`, `omegaconf`, `xarray`, `rotary-embedding-torch`, etc.). Install from the sub-task requirements file:

```bash
# For the baseline GAOT implementation
pip install -r task3/GAOT-base/requirements.txt

# For PyG extensions (replace ${CUDA} with your CUDA version, e.g. cu128 or cpu)
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.7.0+${CUDA}.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.7.0+${CUDA}.html

# The extended implementation uses the same requirements
pip install -r task3/GAOT-random-sampling-dynamic-radius/requirements.txt
```

> **GPU support:** PyTorch is listed without a CUDA suffix in `requirements.txt`. If you need GPU acceleration install a CUDA-enabled wheel from [pytorch.org](https://pytorch.org/get-started/locally/) before running `pip install -r requirements.txt`.

---

## Data

### Task 1

Data is **generated programmatically** by `task1/task1_1_data.py` — no download required. The script samples random coefficients and computes source terms and analytical solutions on a 64×64 grid.

### Task 2

Data consists of NumPy `.npy` trajectory files that are **not included in the repository** (too large). Place them under `data/task2/` with the following layout:

```
data/
└── task2/
    ├── data_train_128.npy              # (1024, 5, 128)  training trajectories
    ├── data_val_128.npy                # (32,   5, 128)  validation trajectories
    ├── data_test_32.npy                # (128,  5,  32)  test at resolution 32
    ├── data_test_64.npy                # (128,  5,  64)  test at resolution 64
    ├── data_test_96.npy                # (128,  5,  96)  test at resolution 96
    ├── data_test_128.npy               # (128,  5, 128)  test at resolution 128
    ├── data_finetune_train_unknown_128.npy  # (32,  5, 128)  fine-tune train
    ├── data_finetune_val_unknown_128.npy    # (8,   5, 128)  fine-tune val
    └── data_test_unknown_128.npy            # (128, 5, 128)  unknown distribution test
```

Each array has shape `(trajectories, time_snapshots, spatial_resolution)` where the five time snapshots correspond to `t ∈ {0.0, 0.25, 0.50, 0.75, 1.0}`.

### Task 3

Task 3 uses NetCDF (`.nc`) datasets from the [GAOT HuggingFace page](https://huggingface.co/datasets/shiwen0710/Datasets_for_GAOT). Place them under `task3/GAOT-base/datasets/time_indep/` (or `time_dep/`) and update the `dataset.base_path` key in the relevant config file. See `task3/GAOT-base/README.md` for details.

---

## Running the Project

All scripts are run from the **project root** unless otherwise noted. Make sure the virtual environment is activated.

### Task 1 — Loss Landscapes: PINNs vs. Data-Driven

The three scripts must be run in order:

**Step 1 — Generate and visualise training data**
```bash
python task1/task1_1_data.py
```
Produces sample plots under `plots/task1_1/` for K = 1, 4, 8, 16.

**Step 2 — Train PINN and Data-Driven solvers**
```bash
python task1/task1_2_train.py
```
Trains MLP solvers for both approaches at three complexity levels (K = 1, 4, 16). Checkpoints are saved to `models/`. Set `USE_PRETRAINED = False` in `task1/config_task1.py` to retrain from scratch; set it to `True` to load existing checkpoints.

**Step 3 — Visualise loss landscapes (bonus)**
```bash
python task1/task1_3_landscape.py
```
Computes 2D loss landscapes around the converged parameters using the filter-normalised direction method (Li et al., 2018). Results are cached in `plots/task1_3/cache/` (HDF5) and plots are written to `plots/task1_3/`.

> **Configuration:** all hyperparameters (grid resolution, K values, landscape range, caching mode, etc.) are centralised in `task1/config_task1.py`.

---

### Task 2 — FNO Dynamical System Identification

The four scripts correspond to the four sub-tasks and should be run in order:

**Step 1 — One-to-one FNO (u₀ → u(t=1))**
```bash
python task2/task1_one2one.py
```

**Step 2 — Resolution generalisation test**
```bash
python task2/task2_resolution.py
```
Evaluates the Task-1 checkpoint at spatial resolutions 32, 64, 96, 128.

**Step 3 — All-to-all time-conditioned FNO**
```bash
python task2/task3_all2all.py
```
Uses all five time snapshots with FiLM-conditioned normalisation.

**Step 4 — Fine-tuning on unknown distribution**
```bash
python task2/task4_finetune.py
```
Fine-tunes the all-to-all model on 32 trajectories from an unknown initial-condition distribution, then evaluates zero-shot and fine-tuned performance.

> **Configuration:** set `USE_PRETRAINED = True` in `task2/config.py` to skip training and load saved checkpoints; set it to `False` to train from scratch. All data and model paths are also configured in that file.

---

### Task 3 — GAOT Extension for Irregular Geometries

Task 3 is self-contained within its subdirectories. Follow the respective READMEs:

**Baseline (Strategy I — Stencil Grid)**
```bash
cd task3/GAOT-base
python main.py --config config/examples/time_indep/elasticity.json
```
See [task3/GAOT-base/README.md](task3/GAOT-base/README.md) for full instructions.

**Extended model (Strategy II — Random Sampling + Dynamic Radius)**
```bash
cd task3/GAOT-random-sampling-dynamic-radius
python main.py --config config/examples/time_indep/elasticity.json
```
To visualise the four tokenization strategies:
```bash
python task3/GAOT-random-sampling-dynamic-radius/visualize_tokenization.py
```
See [task3/GAOT-random-sampling-dynamic-radius/README.md](task3/GAOT-random-sampling-dynamic-radius/README.md) for full instructions.

---

## Results

All generated figures are stored under `plots/` and discussed in detail in `report/report.pdf`.

| Sub-task | What is investigated | Outputs |
|---|---|---|
| **Task 1.1** | Sample visualisations of Poisson source `f` and solution `u` for K = 1, 4, 8, 16 | `plots/task1_1/` |
| **Task 1.2** | Training loss curves and L² relative errors for PINN vs. Data-Driven at K = 1, 4, 16 | `plots/task1_2/` |
| **Task 1.3** | 3D surface and 2D contour loss landscape plots; qualitative comparison of landscape sharpness | `plots/task1_3/` |
| **Task 2.1** | One-to-one FNO: average relative L² error on 128 test trajectories at t = 1.0 | reported in `report.pdf` |
| **Task 2.2** | Resolution generalisation: error across spatial resolutions 32–128 | reported in `report.pdf` |
| **Task 2.3** | All-to-all FNO: error at each time step t = 0.25, 0.50, 0.75, 1.0 | reported in `report.pdf` |
| **Task 2.4** | Zero-shot vs. fine-tuned vs. from-scratch on unknown distribution | reported in `report.pdf` |
| **Task 3** | GAOT baseline vs. extended random-sampling + dynamic-radius model on Elasticity dataset | reported in `report.pdf` |

**Key findings (summary):**
- PINN loss landscapes become significantly rougher and more non-convex as K increases, consistent with Krishnapriyan et al. (2021).
- FNOs generalise well across resolutions (resolution-invariance) due to the spectral parameterisation.
- Fine-tuning with only 32 trajectories substantially closes the gap to in-distribution performance, demonstrating effective transfer learning.
- The random-sampling + dynamic-radius GAOT variant successfully covers irregular domains without leaving coverage holes.

---

## Report

The final submitted report is available at:

```
report/report.pdf
```

The LaTeX source (`report/report.tex`) and bibliography (`report/references.bib`) are also tracked in the repository.

---

## License

**Academic use only — ETH Zurich course project.**

This repository was created as part of the *AI in the Sciences and Engineering (HS25)* course at ETH Zurich. The code is shared for educational and reproducibility purposes. The GAOT implementation in `task3/` is based on the [official GAOT repository](https://github.com/camlab-ethz/GAOT) by Wen et al. (2025, NeurIPS).
