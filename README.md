# AISE 2026 Project: Neural PDE Solvers

A course project for **AI in the Sciences and Engineering (HS25)** at ETH Zurich, by **Marco De Negri**. The project explores three complementary approaches to learning solutions of partial differential equations with neural networks: (1) comparing loss landscapes of Physics-Informed Neural Networks (PINNs) against supervised data-driven solvers on the 2D Poisson equation, (2) training Fourier Neural Operators (FNOs) to identify an unknown dynamical system from trajectory data, and (3) extending the Geometry-Aware Operator Transformer (GAOT) to support irregular-geometry tokenization via random sampling and a dynamic radius strategy.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Prerequisites & Installation](#prerequisites--installation)
3. [Data](#data)
4. [Running the Project](#running-the-project)
5. [Results](#results)
6. [Report](#report)
7. [License](#license)

---

## Project Structure

```
aise2026_project/
│
├── task1/   # Loss landscapes: PINNs vs. Data-Driven — see task1/README.md
├── task2/   # FNO dynamical system identification — see task2/README.md
├── task3/   # GAOT extension for irregular geometries — see task3/README.md
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

Each task has its own README with full run instructions and configuration details. All scripts for Task 1 and Task 2 are run from the project root with the virtual environment activated.

- Task 1: see [task1/README.md](task1/README.md)
- Task 2: see [task2/README.md](task2/README.md)
- Task 3: see [task3/README.md](task3/README.md)

---

## Results

All generated figures are stored under `plots/` and discussed in detail in `report/report.pdf`.

Task 1: On the 2D Poisson equation, the Data-Driven solver outperforms PINN at high frequency (K=16: 0.49% vs. 23.3% relative L² error), while PINN dominates at K=1 (0.0063% vs. 0.13%), consistent with spectral bias theory. Task 2: The all-to-all FNO achieves 0.72% relative L² error across all time steps — a 2.4× improvement over the one-to-one baseline (1.71%) — and fine-tuning on just 32 unknown-distribution trajectories yields 3.07% error versus 9.93% training from scratch. Task 3: The baseline GAOT (4,096 grid tokens) achieves 5.29% relative L¹ error on the Elasticity dataset; the random-sampling variant with dynamic radius (256 tokens) achieves 24.46%, with the performance gap attributed to Transformer positional encodings being designed for regular grids rather than to a coverage deficit.

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
