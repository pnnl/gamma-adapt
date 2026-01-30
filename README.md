# gamma-adapt

## Description

This repository contains the implementation for the research paper "Sim-to-real supervised domain adaptation for radioisotope identification." The project leverages machine learning methods to perform radioisotope identification using gamma spectroscopy. Specifically, this research incorporates domain adaptation techniques to efficiently fuse synthetic data with experimental spectra.

## Installation

### Prerequisites
- Python
- TensorFlow 2.16
- Optuna
- PyRIID (see requirements.txt)

### Steps
```bash
git clone <repository-url>
cd gamma-adapt
pip install -r requirements.txt
```

### Using Docker (Optional)

For containerized deployment:

#### Prerequisites
- Docker and Docker Compose installed
- For GPU support: NVIDIA Docker runtime installed

#### GPU Mode
```bash
docker compose up -d gamma-adapt-gpu
docker exec -it gamma-adapt-gpu bash
```

#### CPU Mode
```bash
docker compose up -d gamma-adapt-cpu
docker exec -it gamma-adapt-cpu bash
```

## Data Preparation

### Directory Structure

Place your datasets in the `data/` directory with the following structure:

```
data/
├── <source_domain>/
│   ├── ss_train.h5
│   ├── ss_val.h5
│   └── ss_test.h5
├── <target_domain>/
│   ├── ss_train.h5
│   ├── ss_val.h5
│   └── ss_test.h5
└── ...
```

### Using Example Datasets

1. Data from the sim-to-sim adaptation (https://doi.org/10.1016/j.nima.2025.171159) are available at [DATASET_URL_PLACEHOLDER]
2. Download and extract to the `data/` directory following the structure above

## Usage

All training scripts use an intuitive command-line interface with named arguments. Use `--help` with any script to see all available options.

### Quick Reference

**Common Flags:**
- `-s` / `--source-domain`: Source domain name
- `-t` / `--target-domain`: Target domain name
- `-a` / `--architecture`: Model architecture (MLP, CNN, BaselineTBNN, TBNN_*)
- `-r` / `--seed`: Random seed (default: 0)
- `-T` / `--train-time`: Training time in minutes
- `-n` / `--n-trials`: Number of hyperparameter trials
- `-m` / `--mode` or `--method`: Training mode or UDA method

**Available Architectures:**
- `MLP` - Multi-Layer Perceptron
- `CNN` - Convolutional Neural Network
- `BaselineTBNN` - Transformer-Based Neural Network
- `TBNN_linear_gap` - TBNN with linear embedding and global average pooling
- `TBNN_linear_cls` - TBNN with linear embedding and classification readout
- `TBNN_mlp_gap` - TBNN with MLP embedding and global average pooling
- `TBNN_mlp_cls` - TBNN with MLP embedding and classification readout

### Baseline Training

Train a baseline model on source domain data:

```bash
# Search for optimal hyperparameters
python -m run.Baseline.search_hyperparameters \
    --source-domain <source> \
    --architecture <arch> \
    --n-trials <n> \
    --train-time <minutes> \
    [--target-domain <target>]

# Train model with best hyperparameters
python -m run.Baseline.train_models \
    --source-domain <source> \
    --architecture <arch> \
    --seed <seed> \
    --train-time <minutes>
```

**Examples:**

```bash
# Search hyperparameters for MLP on GADRAS data (100 trials, 15 min each)
python -m run.Baseline.search_hyperparameters -s gadras -a MLP -n 100 -T 15

# Train MLP model with seed 0 for 60 minutes and also evaluate on target domain
python -m run.Baseline.train_models -s gadras -a MLP -r 0 -T 60
```

**Parameters:**
- `--target-domain` / `-t`: (Optional) Additional domain for evaluation. If specified, the model will be evaluated on both source and target test sets. Useful for measuring domain gap before attempting domain adaptation.
- `--eval-metric` / `-e`: Evaluation metric to use. Options: `accuracy`, `f1`, `ape`, `cosine`.

**Default Values:**
- `--seed`: 0
- `--train-time`: 60 minutes (training), 15 minutes (search)
- `--eval-metric`: accuracy
- `--n-trials`: 100

### Supervised Domain Adaptation (SDA)

Fine-tune a pretrained baseline model using target domain data:

```bash
# Search for optimal hyperparameters
python -m run.SDA.search_hyperparameters \
    --source-domain <source> \
    --target-domain <target> \
    --architecture <arch> \
    --n-trials <n> \
    --train-time <minutes>

# Train model with best hyperparameters
python -m run.SDA.train_models \
    --source-domain <source> \
    --target-domain <target> \
    --architecture <arch> \
    --seed <seed> \
    --train-time <minutes>
```

**Examples:**

```bash
# Fine-tune MLP model using target domain data
python -m run.SDA.train_models \
    -s gadras -t geant4 -a MLP -r 0 -T 60
```

**Parameters:**
- `--training-size` / `-z`: (Optional) Number of target domain samples to use. If not specified, uses the entire dataset.
- `--eval-metric` / `-e`: Evaluation metric to use. Options: `accuracy`, `f1`, `ape`, `cosine`, `crossentropy`.

**Default Values:**
- `--mode`: finetune
- `--eval-metric`: accuracy
- `--seed`: 0
- `--train-time`: 60 minutes

### Unsupervised Domain Adaptation (UDA)

Train using unlabeled target domain data with various UDA methods:

```bash
# Search for optimal hyperparameters
python -m run.UDA.search_hyperparameters \
    --method <method> \
    --source-domain <source> \
    --target-domain <target> \
    --architecture <arch> \
    --n-trials <n> \
    --train-time <minutes>

# Train model with best hyperparameters
python -m run.UDA.train_models \
    --method <method> \
    --source-domain <source> \
    --target-domain <target> \
    --architecture <arch> \
    --seed <seed> \
    --train-time <minutes>
```

**Available Methods:**
- `ADDA` - Adversarial Discriminative Domain Adaptation
- `DAN` - Deep Adaptation Networks
- `DANN` - Domain-Adversarial Neural Networks
- `DeepCORAL` - Deep CORAL
- `DeepJDOT` - Deep Joint Distribution Optimal Transport
- `MeanTeacher` - Mean Teacher
- `SimCLR` - Simple Framework for Contrastive Learning

**Examples:**

```bash
# Search hyperparameters for DAN method with MLP
python -m run.UDA.search_hyperparameters \
    -m DAN -s gadras -t geant4 -a MLP -n 100 -T 15

# Train DAN model
python -m run.UDA.train_models \
    -m DAN -s gadras -t geant4 -a MLP -r 0 -T 120
```

**Parameters:**
- `--eval-metric` / `-e`: Evaluation metric to use. Options: `accuracy`, `f1`, `ape`, `cosine`, `crossentropy`.

**Default Values:**
- `--seed`: 0
- `--eval-metric`: accuracy
- `--train-time`: 120 minutes (training), 15 minutes (search)
- `--n-trials`: 100

## Output Structure

Training outputs are saved to the `out/` directory with clear organization showing domain transfer:

```
out/
├── Baseline/
│   ├── <source_domain>/
│   │   └── <architecture>/
│   │       ├── optuna_study.db
│   │       ├── models.csv
│   │       └── model_<seed>.json
│   └── ...
├── SDA/
│   ├── finetune/
│   │   └── <source>_to_<target>/
│   │       └── <architecture>/
│   │           ├── optuna_study.db
│   │           ├── models.csv
│   │           └── model_<size>_<seed>.json
│   └── fromscratch/
│       └── <source>_to_<target>/
│           └── ...
└── UDA/
    ├── ADDA/
    │   └── <source>_to_<target>/
    │       └── <architecture>/
    │           ├── optuna_study.db
    │           ├── models.csv
    │           └── model_<seed>.json
    ├── DANN/
    └── ...
```

## Environment Variables

Customize paths using environment variables:

- `DATA_DIR`: Path to data directory (default: `./data`)
- `OUT_DIR`: Path to output directory (default: `./out`)

**Example:**
```bash
export DATA_DIR=/mnt/shared/datasets
export OUT_DIR=/mnt/shared/results
python -m run.Baseline.train_models -s gadras -a MLP -T 60
```

## Dependencies

This project requires PyRIID, a Python package for radioisotope identification. See `requirements.txt` for the complete list of dependencies.

Key dependencies:
- TensorFlow 2.16.2
- PyRIID
- Optuna (for hyperparameter optimization)
- NumPy, Pandas, H5Py

## Citation

```
P. Lalor, H. Adams, A. Hagen, Sim-to-real supervised domain adaptation for radioisotope identification,
Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors
and Associated Equipment (2025) 171159 doi:https://doi.org/10.1016/j.nima.2025.171159.
```

## License

This project is licensed under the BSD 2-Clause License. See the LICENSE file for details.

Copyright Battelle Memorial Institute 2026

## Disclaimer

This material was prepared as an account of work sponsored by an agency of the United States Government. See the disclaimer.txt file for full details.

This software is operated by Battelle Memorial Institute for the United States Department of Energy under Contract DE-AC05-76RL01830.
