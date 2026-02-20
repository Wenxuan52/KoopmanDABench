# Contributing Guide

Thanks for your interest in contributing to **Koopman DA Benchmark**.  
This guide explains how to add new **baselines** and/or **datasets** in a way that remains consistent with the benchmark design.

---

## General Principles

- Keep **dataset-level DA scripts unified** under `src/assimilation/`.
- Keep **baseline implementations modular** under `src/models/{model}/{dataset}/`.
- Avoid duplicating evaluation logic across baselines.
- Prefer **config-driven experiments** under `config/`, except when a baseline must stay faithful to its original implementation (e.g., CGKN/DBF).

---

## Development Setup

1. Create an environment:

```bash
conda env create -f environment.yaml
conda activate koopmanda
````

---

## Adding a New Baseline Model

### 1) Create the folder layout

Create:

```text
src/models/<NewModel>/<DATASET_NAME>/
```

Example:

```text
src/models/MyKoopmanModel/ERA5/
```

### 2) Provide training code

Add a trainer script:

```text
src/models/<NewModel>/<DATASET_NAME>/<dataset>_trainer.py
```

Example:

```text
src/models/MyKoopmanModel/ERA5/era5_trainer.py
```

**Requirements**

* The trainer should be runnable by Python (CLI or `main()` function).
* It should save checkpoints/logs in a consistent way (e.g., under an `outputs/` directory or a user-provided path).
* If it uses configs, add a config file under `config/`.

### 3) Provide DA code with a unified interface

Add a dataset DA implementation:

```text
src/models/<NewModel>/<DATASET_NAME>/<dataset>_DA.py
```

Example:

```text
src/models/MyKoopmanModel/ERA5/era5_DA.py
```

**Recommended interface**

* Implement a function named:

```python
run_multi_da_experiment(...)
```

This function is expected to run repeated DA experiments (commonly 5 runs) under the same observation setting (density/noise/locations), and report aggregated statistics (mean/std).
Exact signature can follow existing baselines in the repository (ERA5 DA implementations are a good reference).

> Note: Some baselines (e.g., DBF, CGKN) may keep DA logic close to their original repositories. This is acceptable if it improves faithfulness and reproducibility.

### 4) Register / dispatch support (if needed)

* `src/main.py` should be able to dispatch your baseline trainer from:
  `--dataset <DATASET_NAME> --model <NewModel>`

If the dispatcher uses a fixed mapping table, update it accordingly.
If it discovers trainers by folder convention, ensure your trainer naming matches the convention.

---

## Adding a New Dataset

### 1) Add dataset preparation scripts (recommended)

* Put scripts under `data/` (or `scripts/`) to download/prepare the dataset.
* Do NOT commit large data files into Git.
* Prefer hosting prepared data on Hugging Face and document the link in `README.md`.

### 2) Add dataset-level unified assimilation scripts

Create two scripts in `src/assimilation/`:

```text
src/assimilation/<dataset>_full_observation.py
src/assimilation/<dataset>_intermittent_observation.py
```

These scripts should:

* parse common DA arguments (window length, observation density/noise, interval, number of runs, seeds, etc.)
* call the baseline’s dataset DA implementation (e.g., `run_multi_da_experiment`)

### 3) Add baseline implementations for this dataset

For each baseline to support the new dataset, add:

```text
src/models/<Model>/<DATASET_NAME>/<dataset>_trainer.py
src/models/<Model>/<DATASET_NAME>/<dataset>_DA.py
```

### 4) Add configs

Add dataset configs under:

```text
config/<dataset>/
```

If a baseline is special (like CGKN/DBF), document that its parameters are set inside code instead of config.

---

## Configuration Policy

* Prefer config files under `config/` for training and evaluation.
* If a baseline must follow its upstream implementation strictly (e.g., CGKN/DBF):

  * parameters can be hard-coded in the trainer
  * document where they live and how to change them

---

## Code Style & Quality

* Keep imports explicit and avoid fragile relative-path assumptions.
* Avoid hard-coding absolute paths.
* Keep logging and output directories consistent across baselines when possible.
* Add minimal comments for tricky parts (especially DA scheduling, observation operators, and evaluation metrics).

If CI/linting tools are added later, please ensure your contribution passes them.

---

## Pull Request Checklist

Before opening a PR, please ensure:

* [ ] Folder structure follows the benchmark conventions.
* [ ] Trainer script runs end-to-end (at least on a small example).
* [ ] DA script implements the recommended interface (or explains why it diverges).
* [ ] New configs are added under `config/` (unless baseline is code-configured).
* [ ] README is updated if you add a new dataset/baseline.
* [ ] Any third-party code usage complies with its original license.

---

## Questions

If you are unsure about how to integrate a baseline/dataset, open an issue with:

* baseline/dataset name
* expected training entrypoint
* expected DA task definition
* any special dependencies or licensing constraints
