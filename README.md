# Koopman Data Assimilation Benchmark

A benchmark framework for **Data Assimilation (DA)** tasks with **Koopman-inspired models**.

This repository aims to provide a clean, extensible, and reproducible codebase for:
- training Koopman-inspired baselines on multiple datasets, and
- evaluating them under unified **data assimilation protocols** (single-window DA and intermittent DA).

---

## Key Features

- **Unified benchmark structure**: consistent folder layout across models and datasets.
- **Centralized DA scripts**: dataset-level DA tasks are dispatched through `src/assimilation/`, minimizing duplicated code.
- **Multi-run statistics**: DA code supports repeated experiments (e.g., 5 runs) to report mean/std under the same observation settings.
- **Extensible baselines and datasets**: add new components by following the contribution guide.

---

## Repository Structure

```text
.
├── src/
│   ├── main.py                 # unified training entry
│   ├── models/                 # baselines (training + DA implementations)
│   ├── assimilation/           # dataset-level unified DA task scripts
│   │   ├── <dataset>_full_observation.py
│   │   └── <dataset>_intermittent_observation.py
│   └── plot/                   # visualization scripts
├── config/                     # experiment configs
├── data/                       # local data directory (Only for generation)
├── README.md
└── CONTRIBUTING.md
````

---

## Installation

### Option A: Conda (recommended)

```bash
conda env create -f environment.yaml
conda activate koopman-da
```

### Option B: pip

```bash
pip install -r requirements.txt
```

> Tip: This project is commonly used with CUDA-enabled PyTorch. Make sure your NVIDIA driver is compatible with your CUDA runtime / PyTorch wheel version.

---

## Data

The `data/` folder is intentionally kept **out of Git** due to dataset size.

* We plan to host prepared datasets on **Hugging Face**.
* This repository currently provides:

  * scripts for **crawling / preparing ERA5 High** data
  * scripts for generating **Kolmogorov** data

Please check the `data/` folder (and related scripts) for the latest dataset preparation workflow.

---

## Configuration

All training-related configurations live under `config/`.

* Most baselines read their hyperparameters from config files.
* **CGKN** and **DBF** are special: their parameters are **defined directly in their trainer code** (following their original implementations).

  * If you need to tune CGKN/DBF hyperparameters, please edit the corresponding training scripts under:
    `src/models/CGKN/<DATASET>/...` and `src/models/DBF/<DATASET>/...`.

---

## Training

Use the unified entry:

```bash
python src/main.py --dataset <dataset> --model <model>
```

Example:

```bash
python src/main.py --dataset ERA5 --model KKR
```

`src/main.py` will dispatch to:

```text
src/models/<MODEL>/<DATASET>/<dataset>_trainer.py
```

Additional CLI arguments (if supported by the trainer) will be forwarded to the underlying trainer script.

---

## Data Assimilation Benchmark

DA runs are dispatched through `src/assimilation/`.

For each dataset, two standard tasks are provided:

1. **Full / single-window assimilation**
   Fixed assimilation window length, single DA run.

2. **Intermittent / repeated assimilation**
   Fixed window length + assimilation interval (multiple DA updates).

Baseline DA implementations (e.g., ERA5) typically provide a unified interface:

* `src/models/{model_name}/ERA5/era5_DA.py`

  * includes `run_multi_da_experiment(...)` for repeated runs and reporting mean/std

The dataset-level scripts in `src/assimilation/` handle unified scheduling and argument passing so you can run many experiments with minimal code changes.

> For exact CLI usage and available arguments, please open the corresponding scripts in `src/assimilation/` and the dataset DA file under `src/models/.../..._DA.py`.

---

## Adding a New Dataset or Baseline

If you want to add a new dataset and/or baseline model, please read:

* [`CONTRIBUTING.md`](CONTRIBUTING.md)

It documents the expected folder structure, minimal interfaces (trainer + DA), and how to register new components into the benchmark workflow.

---

## Citation

If you use this benchmark in your research, please cite it (see `CITATION.cff`).

---

## Acknowledgements

This benchmark includes baseline implementations inspired by Koopman operator learning and related approaches.
We thank the authors of the original repositories and papers.
