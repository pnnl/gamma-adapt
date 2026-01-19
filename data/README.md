# Data Directory

This directory contains the datasets used for training and evaluating radioisotope identification models.

## Directory Structure

The expected directory structure is simple and flexible:

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

## HDF5 File Format

All datasets are stored in HDF5 format compatible with PyRIID's `SampleSet` class.

## Using Custom Datasets

To use your own datasets with this codebase:

### 1. Prepare Your Data

Format your data as HDF5 files compatible with PyRIID's `SampleSet` class:

### 2. Organize Directory Structure

Place your HDF5 files in the data directory with any domain name:

```
data/
└── my_custom_domain/
    ├── ss_train.h5
    ├── ss_val.h5
    └── ss_test.h5
```

### 3. Use in Training Scripts

Reference your domain name when running training:

```bash
# Baseline training
python -m run.Baseline.train_models -s my_custom_domain -a MLP -T 60

# SDA fine-tuning from one domain to another
python -m run.SDA.train_models -m finetune -s source_domain -t my_custom_domain -a CNN -T 60

# UDA training
python -m run.UDA.train_models -m DANN -s source_domain -t my_custom_domain -a MLP -T 120
```

## Example Datasets

Example datasets for this project can be downloaded from:

**[DATASET_URL_PLACEHOLDER - To be provided]**

After downloading, extract the datasets to maintain the directory structure described below.