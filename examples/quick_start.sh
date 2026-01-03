#!/bin/bash
# Quick Start Example - Complete workflow demonstrating all training modes
#
# This script demonstrates a typical workflow:
# 1. Search for optimal baseline hyperparameters
# 2. Train a baseline model on source domain
# 3. Search for optimal SDA hyperparameters
# 4. Fine-tune the model on target domain (SDA)
# 5. Search for optimal UDA hyperparameters
# 6. Train with unsupervised domain adaptation (UDA)
#
# Usage: bash examples/quick_start.sh

set -e  # Exit on error

# Configuration
SOURCE_DOMAIN="sim2real_gadras"
TARGET_DOMAIN="sim2real_experimental"
ARCHITECTURE="MLP"
UDA_METHOD="DeepCORAL"
EVAL_METRIC="accuracy"
SEED=0
N_TRIALS=3
SEARCH_TIME=1  # minutes per trial for hyperparameter search
TRAIN_TIME=1   # minutes for model training

echo "============================================"
echo "Quick Start Example"
echo "============================================"
echo "Source domain: $SOURCE_DOMAIN"
echo "Target domain: $TARGET_DOMAIN"
echo "Architecture:  $ARCHITECTURE"
echo "UDA Method:    $UDA_METHOD"
echo "Eval Metric:   $EVAL_METRIC"
echo "N Trials:      $N_TRIALS"
echo "Search Time:   $SEARCH_TIME min"
echo "Train Time:    $TRAIN_TIME min"
echo "============================================"
echo

# Step 1: Search for optimal baseline hyperparameters
echo "Step 1/6: Searching for optimal baseline hyperparameters..."
python -m run.Baseline.search_hyperparameters \
    --source-domain $SOURCE_DOMAIN \
    --architecture $ARCHITECTURE \
    --n-trials $N_TRIALS \
    --train-time $SEARCH_TIME \
    --eval-metric $EVAL_METRIC

echo
echo "✓ Baseline hyperparameter search complete!"
echo

# Step 2: Train baseline model
echo "Step 2/6: Training baseline model..."
python -m run.Baseline.train_models \
    --source-domain $SOURCE_DOMAIN \
    --architecture $ARCHITECTURE \
    --seed $SEED \
    --train-time $TRAIN_TIME \
    --target-domain $TARGET_DOMAIN \
    --eval-metric $EVAL_METRIC

echo
echo "✓ Baseline training complete!"
echo

# Step 3: Search for optimal SDA hyperparameters
echo "Step 3/6: Searching for optimal SDA hyperparameters..."
python -m run.SDA.search_hyperparameters \
    --source-domain $SOURCE_DOMAIN \
    --target-domain $TARGET_DOMAIN \
    --architecture $ARCHITECTURE \
    --n-trials $N_TRIALS \
    --train-time $SEARCH_TIME \
    --eval-metric $EVAL_METRIC

echo
echo "✓ SDA hyperparameter search complete!"
echo

# Step 4: Fine-tune on target domain (Supervised Domain Adaptation)
echo "Step 4/6: Fine-tuning on target domain (SDA)..."
python -m run.SDA.train_models \
    --source-domain $SOURCE_DOMAIN \
    --target-domain $TARGET_DOMAIN \
    --architecture $ARCHITECTURE \
    --seed $SEED \
    --train-time $TRAIN_TIME \
    --eval-metric $EVAL_METRIC

echo
echo "✓ SDA fine-tuning complete!"
echo

# Step 5: Search for optimal UDA hyperparameters
echo "Step 5/6: Searching for optimal UDA hyperparameters..."
python -m run.UDA.search_hyperparameters \
    --method $UDA_METHOD \
    --source-domain $SOURCE_DOMAIN \
    --target-domain $TARGET_DOMAIN \
    --architecture $ARCHITECTURE \
    --n-trials $N_TRIALS \
    --train-time $SEARCH_TIME \
    --eval-metric $EVAL_METRIC

echo
echo "✓ UDA hyperparameter search complete!"
echo

# Step 6: Train with Unsupervised Domain Adaptation
echo "Step 6/6: Training with Unsupervised Domain Adaptation ($UDA_METHOD)..."
python -m run.UDA.train_models \
    --method $UDA_METHOD \
    --source-domain $SOURCE_DOMAIN \
    --target-domain $TARGET_DOMAIN \
    --architecture $ARCHITECTURE \
    --seed $SEED \
    --train-time $TRAIN_TIME \
    --eval-metric $EVAL_METRIC

echo
echo "============================================"
echo "✓ All training complete!"
echo "============================================"
echo

# Display results
echo "============================================"
echo "RESULTS SUMMARY"
echo "============================================"
echo

echo "Baseline Model (trained on $SOURCE_DOMAIN):"
echo "File: out/Baseline/$SOURCE_DOMAIN/$ARCHITECTURE/models.csv"
echo "-------------------------------------------"
cat "out/Baseline/$SOURCE_DOMAIN/$ARCHITECTURE/models.csv"
echo

echo "SDA Model (fine-tuned on $TARGET_DOMAIN):"
echo "File: out/SDA/finetune/${SOURCE_DOMAIN}_to_${TARGET_DOMAIN}/$ARCHITECTURE/models.csv"
echo "-------------------------------------------"
cat "out/SDA/finetune/${SOURCE_DOMAIN}_to_${TARGET_DOMAIN}/$ARCHITECTURE/models.csv"
echo

echo "UDA Model ($UDA_METHOD: $SOURCE_DOMAIN → $TARGET_DOMAIN):"
echo "File: out/UDA/$UDA_METHOD/${SOURCE_DOMAIN}_to_${TARGET_DOMAIN}/$ARCHITECTURE/models.csv"
echo "-------------------------------------------"
cat "out/UDA/$UDA_METHOD/${SOURCE_DOMAIN}_to_${TARGET_DOMAIN}/$ARCHITECTURE/models.csv"
echo

echo "============================================"
