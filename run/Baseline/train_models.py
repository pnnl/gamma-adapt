import numpy as np
import tensorflow as tf
import csv
import sys
import os
import argparse
from riid import read_hdf
from riid.models import MLP, CNN, TBNN, BaselineTBNN
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.losses import CategoricalCrossentropy as CategoricalCrossentropyLoss
import random
from ..utils.hyperparameter_utils import load_best_baseline_hyperparameters

# Get configurable paths from environment variables
DATA_DIR = os.environ.get('DATA_DIR', 'data')
OUT_DIR = os.environ.get('OUT_DIR', 'out')

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='Train a baseline model on source domain data',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--source-domain', '-s', required=True,
                    help='Name of source domain (e.g., "gadras", "sim_v1")')
parser.add_argument('--architecture', '-a', required=True,
                    choices=['MLP', 'CNN', 'BaselineTBNN', 'TBNN_linear_cls', 'TBNN_linear_gap',
                             'TBNN_mlp_cls', 'TBNN_cnn_cls'],
                    help='Model architecture to train')
parser.add_argument('--target-domain', '-t', default=None,
                    help='Name of target domain for additional evaluation (optional)')
parser.add_argument('--seed', '-r', type=int, default=0,
                    help='Random seed for reproducibility')
parser.add_argument('--train-time', '-T', type=float, default=60,
                    help='Training time in minutes')
parser.add_argument('--eval-metric', '-e', default='accuracy',
                    choices=['accuracy', 'ape', 'f1', 'cosine', 'crossentropy'],
                    help='Evaluation metric to use (in addition to crossentropy loss)')

args = parser.parse_args()

source_domain = args.source_domain
architecture = args.architecture
target_domain = args.target_domain
seed = args.seed
train_time = args.train_time * 60
eval_metric = args.eval_metric

# Map metric names to PyRIID method names
metric_methods = {
    'accuracy': 'calc_accuracy',
    'ape': 'calc_APE_score',
    'f1': 'calc_f1_score',
    'cosine': 'calc_cosine_similarity',
    'crossentropy': 'calc_crossentropy'
}

print(f"\n{'='*60}")
print(f"Baseline Training Configuration")
print(f"{'='*60}")
print(f"Source domain:    {source_domain}")
if target_domain:
    print(f"Target domain:    {target_domain}")
print(f"Architecture:     {architecture}")
print(f"Eval metric:      {eval_metric}")
print(f"Random seed:      {seed}")
print(f"Training time:    {args.train_time} minutes")
print(f"{'='*60}\n")

random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

training_ss = read_hdf(os.path.join(DATA_DIR, f"{source_domain}/ss_train.h5"))
validation_ss = read_hdf(os.path.join(DATA_DIR, f"{source_domain}/ss_val.h5"))
source_testing_ss = read_hdf(os.path.join(DATA_DIR, f"{source_domain}/ss_test.h5"))

# Load target test data only if specified
if target_domain:
    target_testing_ss = read_hdf(os.path.join(DATA_DIR, f"{target_domain}/ss_test.h5"))
else:
    target_testing_ss = None

hyperparams = load_best_baseline_hyperparameters(architecture, source_domain)

print(f"Loaded hyperparameters for {architecture} model on {source_domain} dataset:")
for key, value in hyperparams.items():
    print(f"    {key} = {value}")

learning_rate = hyperparams["learning_rate"]
weight_decay = hyperparams["weight_decay"]
batch_size = hyperparams["batch_size"]
dropout = hyperparams["dropout"]
label_smoothing = hyperparams["label_smoothing"]

optimizer = AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
loss = CategoricalCrossentropyLoss(label_smoothing=label_smoothing)

if architecture == "MLP":
    model = MLP(
        optimizer=optimizer,
        loss=loss,
        hidden_layers=hyperparams["hidden_layers"],
        dropout=dropout,
        normalize="sqrt_zscore"
    )
elif architecture == "CNN":
    model = CNN(
        optimizer=optimizer,
        loss=loss,
        filters=hyperparams["filters"],
        kernel_sizes=hyperparams["kernel_sizes"],
        dense_layer_sizes=hyperparams["dense_layer_sizes"],
        dropout=dropout,
        normalize="sqrt_zscore"
    )
elif architecture == "BaselineTBNN":
    model = BaselineTBNN(
        optimizer=optimizer,
        loss=loss,
        num_heads=hyperparams["num_heads"],
        ff_dim=hyperparams["ff_dim"],
        num_layers=hyperparams["num_layers"],
        dropout=dropout,
        normalize="sqrt_zscore"
    )
elif architecture.startswith("TBNN"):
    arch, embed_mode, readout = architecture.split("_")
    if embed_mode in ["mlp", "cnn"]:
        embed_inner = hyperparams["embed_inner"]
    else:
        embed_inner = None
    model = TBNN(
        optimizer=optimizer,
        loss=loss,
        embed_mode=embed_mode,
        embed_inner=embed_inner,
        embed_dim=hyperparams["embed_dim"],
        pos_encoding="learnable",
        num_heads=hyperparams["num_heads"],
        ff_dim=hyperparams["ff_dim"],
        num_layers=hyperparams["num_layers"],
        patch_size=hyperparams["patch_size"],
        stride=hyperparams["stride"],
        readout=readout,
        dropout=dropout,
        normalize="sqrt_zscore"
    )
else:
    raise ValueError(f"Unknown architecture: {architecture}")

model.fit(training_ss, validation_ss, batch_size=batch_size,
          patience=40, verbose=True, training_time=train_time,
          es_monitor="val_loss")

outdir = os.path.join(OUT_DIR, f"Baseline/{source_domain}/{architecture}")
os.makedirs(outdir, exist_ok=True)

model_filename = os.path.join(outdir, f"model_{seed}.json")
model.save(model_filename, overwrite=True)
print(f"Saved model to {model_filename}")

# Always evaluate on source domain test set
source_crossentropy = model.calc_crossentropy(source_testing_ss)
calc_metric = getattr(model, metric_methods[eval_metric])
source_eval_score = calc_metric(source_testing_ss)

print(f"For {architecture} model trained on {source_domain}:")
print(f"    {source_domain}_crossentropy = {source_crossentropy:.4g}")
print(f"    {source_domain}_{eval_metric} = {source_eval_score:.4g}")

# Evaluate on target domain if specified
if target_testing_ss is not None:
    target_crossentropy = model.calc_crossentropy(target_testing_ss)
    target_eval_score = calc_metric(target_testing_ss)
    print(f"    {target_domain}_crossentropy = {target_crossentropy:.4g}")
    print(f"    {target_domain}_{eval_metric} = {target_eval_score:.4g}")

# Write results to CSV
output_filename = os.path.join(outdir, "models.csv")
file_exists = os.path.exists(output_filename)
with open(output_filename, "a", newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
        if target_testing_ss is not None:
            # Both source and target metrics
            writer.writerow(["seed", f"{source_domain}_crossentropy", f"{source_domain}_{eval_metric}", 
                           f"{target_domain}_crossentropy", f"{target_domain}_{eval_metric}"])
        else:
            # Only source metrics
            writer.writerow(["seed", f"{source_domain}_crossentropy", f"{source_domain}_{eval_metric}"])
    
    if target_testing_ss is not None:
        writer.writerow([seed, source_crossentropy, source_eval_score, target_crossentropy, target_eval_score])
    else:
        writer.writerow([seed, source_crossentropy, source_eval_score])
    print(f"Saved output to {output_filename}")
