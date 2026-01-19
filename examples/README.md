# Usage Examples

This directory contains example scripts and workflows for using gamma-adapt.

## Files

- **quick_start.sh** - Complete end-to-end workflow example (hyperparameter search → baseline training → fine-tuning)
- **README.md** - This file with detailed usage examples

## Quick Start with Docker

### 1. Build and Start Container

**For GPU:**
```bash
docker-compose up -d gamma-adapt-gpu
docker exec -it gamma-adapt-gpu bash
```

**For CPU:**
```bash
docker-compose up -d gamma-adapt-cpu
docker exec -it gamma-adapt-cpu bash
```

### 2. Prepare Data

Download the official datasets from [DATASET_URL_PLACEHOLDER] and place them in the `data/` directory with the structure:

```
data/
├── gadras/
│   ├── ss_train.h5
│   ├── ss_val.h5
│   └── ss_test.h5
└── geant4/
    ├── ss_train.h5
    ├── ss_val.h5
    └── ss_test.h5
```

Or mount your own data directory using docker-compose.

### 3. Run the Quick Start Script

Once your data is ready, you can run the complete workflow:

```bash
# Run the workflow
bash examples/quick_start.sh
```

This script will:
1. Search for optimal baseline hyperparameters on GADRAS data
2. Train a baseline MLP model
3. Search for optimal SDA hyperparameters
4. Fine-tune the model on Geant4 data (SDA)
5. Search for optimal UDA hyperparameters
6. Train with unsupervised domain adaptation (UDA using DAN)

Results will be saved to the `out/` directory and metrics will be displayed during training.

## Example Workflows

### Baseline Training

#### 1. Hyperparameter Search

Search for optimal hyperparameters for an MLP model on GADRAS synthetic data:

```bash
python -m run.Baseline.search_hyperparameters \
    --source-domain gadras \
    --architecture MLP \
    --n-trials 100 \
    --train-time 15

# Or with short flags
python -m run.Baseline.search_hyperparameters -s gadras -a MLP -n 100 -T 15
```

This runs 100 Optuna trials, with each trial training for up to 15 minutes.

#### 2. Train Models

Once hyperparameters are found, train models:

```bash
# Train on GADRAS, include evaluation metrics on Geant4
python -m run.Baseline.train_models -s gadras -a MLP -r 0 -T 60 -t geant4
```

#### 3. Available Architectures

- `MLP` - Multi-layer Perceptron
- `CNN` - Convolutional Neural Network
- `BaselineTBNN` - Baseline Transformer-based Neural Network
- `TBNN_linear_cls` - TBNN with linear embedding and CLS token readout
- `TBNN_linear_gap` - TBNN with linear embedding and global average pooling
- `TBNN_mlp_cls` - TBNN with MLP embedding
- `TBNN_cnn_cls` - TBNN with CNN embedding

### Supervised Domain Adaptation (SDA)

Fine-tune a pretrained model on target domain data:

```bash
# Using full target dataset (recommended)
python -m run.SDA.train_models \
    --source-domain gadras \
    --target-domain geant4 \
    --architecture CNN \
    --seed 0 \
    --train-time 60

# Or with short flags
python -m run.SDA.train_models -s gadras -t geant4 -a CNN -r 0 -T 60
```

### Unsupervised Domain Adaptation (UDA)

Train models using unlabeled target domain data with various UDA methods:

#### DAN (Deep Adaptation Networks)

```bash
# Train DANN model
python -m run.UDA.train_models \
    --method DAN \
    --source-domain gadras \
    --target-domain geant4 \
    --architecture MLP \
    --seed 0 \
    --train-time 120

# Or with short flags
python -m run.UDA.train_models -m DAN -s gadras -t geant4 -a MLP -r 0 -T 120
```

#### Other UDA Methods

Available methods:
- `ADDA` - Adversarial Discriminative Domain Adaptation
- `DAN` - Deep Adaptation Networks
- `DANN` - Domain-Adversarial Neural Networks
- `DeepCORAL` - Deep CORAL
- `DeepJDOT` - Deep Joint Distribution Optimal Transport
- `MeanTeacher` - Mean Teacher
- `SimCLR` - Simple Framework for Contrastive Learning

## Environment Variables

Customize data and output paths:

```bash
export DATA_DIR=/path/to/my/data
export OUT_DIR=/path/to/my/outputs

# Then run training normally
python -m run.Baseline.train_models -s gadras -a MLP -T 60
```

## Monitoring Training

### View Outputs

Results are saved to the `out/` directory:

```bash
# View training results
cat out/Baseline/gadras/MLP/models.csv

# View SDA results
cat out/SDA/finetune/gadras_to_geant4/MLP/models.csv

# View UDA results
cat out/UDA/DAN/gadras_to_geant4/MLP/models.csv
```

## Batch Training Examples

### Train Multiple Seeds

```bash
# Train baseline models with 5 different seeds
for seed in 0 1 2 3 4; do
    python -m run.Baseline.train_models -s gadras -a MLP -r $seed -T 60
done
```

### Train Multiple Architectures

```bash
# Train all architectures
for arch in MLP CNN TBNN_linear_gap; do
    python -m run.Baseline.train_models -s gadras -a $arch -r 0 -T 60
done
```

### Data Efficiency Experiments

```bash
# Train with different sample sizes
for size in 256 512 1024 2048 4096; do
    python -m run.SDA.train_models -s gadras -t geant4 -a CNN -z $size -r 0 -T 60
done
```
