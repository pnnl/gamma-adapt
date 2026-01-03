import numpy as np
import tensorflow as tf
import csv
import sys
import os
import argparse
import math
from riid import read_hdf, concat_ss
from riid.models import MLP, CNN, TBNN, BaselineTBNN
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.losses import CategoricalCrossentropy as CategoricalCrossentropyLoss
from tensorflow.keras.metrics import CategoricalCrossentropy
from tensorflow.keras.layers import Dropout
import random
from ..utils.hyperparameter_utils import load_best_sda_finetune_hyperparameters, load_best_sda_fromscratch_hyperparameters, load_best_baseline_hyperparameters

# Get configurable paths from environment variables
DATA_DIR = os.environ.get('DATA_DIR', 'data')
OUT_DIR = os.environ.get('OUT_DIR', 'out')

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='Train a Supervised Domain Adaptation (SDA) model',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--mode', '-m', default='finetune',
                    choices=['fromscratch', 'finetune'],
                    help='Training mode: finetune (fine-tune pretrained model) or fromscratch (train from scratch)')
parser.add_argument('--architecture', '-a', required=True,
                    choices=['MLP', 'CNN', 'BaselineTBNN', 'TBNN_linear_cls', 'TBNN_linear_gap',
                             'TBNN_mlp_cls', 'TBNN_cnn_cls'],
                    help='Model architecture to train')
parser.add_argument('--source-domain', '-s', required=True,
                    help='Name of source domain (e.g., "gadras", "sim_v1")')
parser.add_argument('--target-domain', '-t', required=True,
                    help='Name of target domain (e.g., "geant4", "exp_v1")')
parser.add_argument('--training-size', '-z', type=int, default=None,
                    help='Number of training samples from target domain (default: use entire dataset)')
parser.add_argument('--seed', '-r', type=int, default=0,
                    help='Random seed for reproducibility')
parser.add_argument('--train-time', '-T', type=float, default=60,
                    help='Training time in minutes')
parser.add_argument('--eval-metric', '-e', default='accuracy',
                    choices=['accuracy', 'ape', 'f1', 'cosine', 'crossentropy'],
                    help='Evaluation metric to use (in addition to crossentropy loss)')

args = parser.parse_args()

mode = args.mode
architecture = args.architecture
source_domain = args.source_domain
target_domain = args.target_domain
training_size = args.training_size
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

random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# Load data based on whether training_size is specified
if training_size is None:
    # Use entire training dataset
    target_training_ss = read_hdf(os.path.join(DATA_DIR, f"{target_domain}/ss_train.h5"))
    size_label = "full"
    training_size_id = "full"
else:
    # Load first N training samples
    target_training_ss = read_hdf(
        os.path.join(DATA_DIR, f"{target_domain}/ss_train.h5"),
        row_slice=(0, training_size)
    )
    size_label = str(training_size)
    # Use exact training_size for hyperparameter lookup (will fall back to "full" if not found)
    training_size_id = training_size

print(f"\n{'='*60}")
print(f"SDA Training Configuration")
print(f"{'='*60}")
print(f"Mode:             {mode}")
print(f"Source domain:    {source_domain}")
print(f"Target domain:    {target_domain}")
print(f"Architecture:     {architecture}")
print(f"Training size:    {size_label}")
print(f"Eval metric:      {eval_metric}")
print(f"Random seed:      {seed}")
print(f"Training time:    {args.train_time} minutes")
print(f"{'='*60}\n")

target_validation_ss = read_hdf(os.path.join(DATA_DIR, f"{target_domain}/ss_val.h5"))
target_testing_ss = read_hdf(os.path.join(DATA_DIR, f"{target_domain}/ss_test.h5"))

if mode == "fromscratch":
    hyperparams = load_best_sda_fromscratch_hyperparameters(architecture, source_domain, target_domain, training_size_id)
elif mode == "finetune":
    hyperparams = load_best_sda_finetune_hyperparameters(architecture, source_domain, target_domain, training_size_id)

print(f"Loaded hyperparameters for {architecture} model {mode} mode:")
for key, value in hyperparams.items():
    print(f"    {key} = {value}")

learning_rate = hyperparams["learning_rate"]
weight_decay = hyperparams["weight_decay"]
batch_size = hyperparams["batch_size"]
dropout = hyperparams["dropout"]
label_smoothing = hyperparams["label_smoothing"]
optimizer = AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
loss = CategoricalCrossentropyLoss(label_smoothing=label_smoothing)

if mode == "fromscratch":
    baseline_hyperparams = load_best_baseline_hyperparameters(architecture, source_domain)
        
    if architecture == "MLP":
        model = MLP(
            optimizer=optimizer,
            loss=loss,
            hidden_layers=baseline_hyperparams["hidden_layers"],
            dropout=dropout,
            normalize="sqrt_zscore"
        )
    elif architecture == "CNN":
        model = CNN(
            optimizer=optimizer,
            loss=loss,
            filters=baseline_hyperparams["filters"],
            kernel_sizes=baseline_hyperparams["kernel_sizes"],
            dense_layer_sizes=baseline_hyperparams["dense_layer_sizes"],
            dropout=dropout,
            normalize="sqrt_zscore"
        )
    elif architecture == "BaselineTBNN":
        model = BaselineTBNN(
            optimizer=optimizer,
            loss=loss,
            num_heads=baseline_hyperparams["num_heads"],
            ff_dim=baseline_hyperparams["ff_dim"],
            num_layers=baseline_hyperparams["num_layers"],
            dropout=dropout,
            normalize="sqrt_zscore"
        )
    elif architecture.startswith("TBNN"):
        arch, embed_mode, readout = architecture.split("_")
        if embed_mode in ["mlp", "cnn"]:
            embed_inner = baseline_hyperparams["embed_inner"]
        else:
            embed_inner = None
        model = TBNN(
            optimizer=optimizer,
            loss=loss,
            embed_mode=embed_mode,
            embed_inner=embed_inner,
            embed_dim=baseline_hyperparams["embed_dim"],
            pos_encoding="learnable",
            num_heads=baseline_hyperparams["num_heads"],
            ff_dim=baseline_hyperparams["ff_dim"],
            num_layers=baseline_hyperparams["num_layers"],
            patch_size=baseline_hyperparams["patch_size"],
            stride=baseline_hyperparams["stride"],
            readout=readout,
            dropout=dropout,
            normalize="sqrt_zscore"
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
elif mode == "finetune":
    frozen_indices = hyperparams["frozen_indices"]

    if architecture == "MLP":
        model = MLP()
    elif architecture == "CNN":
        model = CNN()
    elif architecture == "BaselineTBNN":
        model = BaselineTBNN()
    elif architecture.startswith("TBNN"):
        model = TBNN()
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    model.load(os.path.join(OUT_DIR, f"Baseline/{source_domain}/{architecture}/model_{seed}.json"))
    pretrained_model = model.model

    for layer in pretrained_model.layers:
        if isinstance(layer, Dropout):
            layer.rate = dropout

    for idx in frozen_indices:
        pretrained_model.layers[idx].trainable = False

    pretrained_model.compile(
        optimizer=optimizer,
        loss=loss
    )

model.fit(target_training_ss, target_validation_ss, batch_size=batch_size,
        patience=10, verbose=True, training_time=train_time, es_monitor="val_loss")

outdir = os.path.join(OUT_DIR, f"SDA/{mode}/{source_domain}_to_{target_domain}/{architecture}")
os.makedirs(outdir, exist_ok=True)

# Create model filename based on whether we used full dataset or partial
if training_size is None:
    model_filename = os.path.join(outdir, f"model_full_{seed}.json")
else:
    model_filename = os.path.join(outdir, f"model_{training_size}_{seed}.json")
model.save(model_filename, overwrite=True)
print(f"Saved model to {model_filename}")

target_crossentropy = model.calc_crossentropy(target_testing_ss)
calc_metric = getattr(model, metric_methods[eval_metric])
target_eval_score = calc_metric(target_testing_ss)

print(f"For {architecture} {mode} model ({source_domain} → {target_domain}):")
print(f"    Training size: {size_label}, Seed: {seed}")
print(f"    {target_domain}_crossentropy = {target_crossentropy:.4g}")
print(f"    {target_domain}_{eval_metric} = {target_eval_score:.4g}")

output_filename = os.path.join(outdir, "models.csv")
file_exists = os.path.exists(output_filename)
with open(output_filename, "a", newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["seed", "training_size", f"{target_domain}_crossentropy", f"{target_domain}_{eval_metric}"])
    writer.writerow([seed, size_label, target_crossentropy, target_eval_score])
    print(f"Saved output to {output_filename}")
