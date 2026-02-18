# Task 3: Extending GAOT for Irregular-Geometry Tokenization

Extends the Geometry-Aware Operator Transformer (GAOT) to support irregular-geometry tokenization via random point sampling and a dynamic radius strategy. The task establishes a baseline with the official GAOT implementation and then introduces random sampling of latent tokens instead of a structured stencil grid.

---

## Directory Structure

```
task3/
├── GAOT-base/                          # Official GAOT implementation (baseline)
│   ├── main.py                         # Entry point — config-driven training and inference
│   ├── requirements.txt
│   ├── config/examples/
│   │   ├── time_indep/                 # elasticity.json, poisson_gauss.json
│   │   └── time_dep/                   # ns_gauss.json, ce_crp.json
│   ├── datasets/time_indep/            # Place .nc dataset files here
│   ├── src/
│   │   ├── core/                       # Base trainer and default config dataclasses
│   │   ├── datasets/                   # Data processors and graph builder
│   │   ├── model/                      # GAOT model and layer definitions
│   │   ├── trainer/                    # StaticTrainer, SequentialTrainer
│   │   └── utils/                      # Metrics, plotting, scaling
│   └── .ckpt / .loss / .results / .database/   # Auto-generated output dirs
│
└── GAOT-random-sampling-dynamic-radius/  # Extended implementation
    ├── main.py
    ├── requirements.txt
    ├── config/examples/                # Config files (same format as base)
    ├── datasets/                       # Dataset files
    ├── visualize_tokenization.py       # Visualise random token placement
    ├── plots/                          # Tokenization visualisation outputs
    └── src/                            # Modified source (random sampling + dynamic radius)
```

---

## Data Setup

Task 3 uses NetCDF (`.nc`) datasets from the [GAOT HuggingFace page](https://huggingface.co/datasets/shiwen0710/Datasets_for_GAOT).

Place datasets under `GAOT-base/datasets/time_indep/` (or `time_dep/`) and set `dataset.base_path` in the config file to point to that folder. For example:

```
GAOT-base/datasets/
└── time_indep/
    ├── Elasticity.nc
    └── Poisson-Gauss.nc
```

---

## Configuration

Experiments are fully driven by JSON (or TOML) config files in `config/examples/`. The most important keys:

| Key | Description |
|---|---|
| `dataset.base_path` | Path to the folder containing `.nc` dataset files |
| `dataset.name` | Dataset name without extension (e.g. `"Elasticity"`) |
| `setup.train` | `true` to train, `false` for inference only |
| `setup.test` | `true` to evaluate after training / during inference |
| `setup.trainer_name` | `"static"` for time-independent, `"sequential"` for time-dependent |
| `path.ckpt_path` | Where to save / load model checkpoints |

For the full list of options and their defaults, see `src/core/default_configs.py`.

---

## Running

All commands are run from inside the respective sub-directory (`GAOT-base/` or `GAOT-random-sampling-dynamic-radius/`).

### Baseline — train on Elasticity

```bash
cd task3/GAOT-base
python main.py --config config/examples/time_indep/elasticity.json
```

### Run all configs in a folder

```bash
python main.py --folder config/examples/time_indep/
```

### Inference only (no retraining)

Set `setup.train: false` and `setup.test: true` in the config, then run:

```bash
python main.py --config config/examples/time_indep/elasticity.json
```

### Extended implementation (random sampling + dynamic radius)

```bash
cd task3/GAOT-random-sampling-dynamic-radius
python main.py --config config/examples/time_indep/elasticity.json
```

To visualise the random token placement:

```bash
python visualize_tokenization.py
```

### Additional CLI flags

| Flag | Description |
|---|---|
| `--debug` | Enable debug mode |
| `--num_works_per_device N` | Parallel data-loader workers per device |
| `--visible_devices 0 1` | Select CUDA devices |

---

## Outputs

Training artifacts are saved automatically to subdirectories relative to the script location:

| Directory | Contents |
|---|---|
| `.ckpt/<problem>/` | Model checkpoints (`.pt`) |
| `.loss/<problem>/` | Loss curves (PNG + NPZ) |
| `.results/<problem>/` | Prediction visualisations |
| `.database/<problem>/` | CSV with error metrics, parameter count, and timing |
