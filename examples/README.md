# Usage Examples

This directory contains example scripts and workflows for using gamma-adapt.

## Files

- **quick_start.sh** - Complete end-to-end workflow example (hyperparameter search → baseline training → fine-tuning)
- **README.md** - This file with detailed usage examples

## Quick Start

### 1. Prepare Data

Download the example datasets from [DATASET_URL_PLACEHOLDER] and place them in the `data/` directory:

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

### 2. Run the Quick Start Script

```bash
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
