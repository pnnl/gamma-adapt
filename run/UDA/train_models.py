import numpy as np
import tensorflow as tf
import csv
import sys
import os
import argparse
from riid import read_hdf
from riid.models import MLP, CNN, TBNN, BaselineTBNN
from riid.models import ADDA, DANN, DAN, DeepCORAL, DeepJDOT, MeanTeacher, SimCLR
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.metrics import CategoricalCrossentropy
import random
from ..utils.hyperparameter_utils import load_best_uda_hyperparameters

# Get configurable paths from environment variables
DATA_DIR = os.environ.get('DATA_DIR', 'data')
OUT_DIR = os.environ.get('OUT_DIR', 'out')

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='Train an Unsupervised Domain Adaptation (UDA) model',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--method', '-m', required=True,
                    choices=['ADDA', 'DANN', 'DAN', 'DeepCORAL', 'DeepJDOT', 'MeanTeacher', 'SimCLR'],
                    help='UDA method to use')
parser.add_argument('--architecture', '-a', required=True,
                    choices=['MLP', 'CNN', 'BaselineTBNN', 'TBNN_linear_cls', 'TBNN_linear_gap',
                             'TBNN_mlp_cls', 'TBNN_cnn_cls'],
                    help='Model architecture to train')
parser.add_argument('--source-domain', '-s', required=True,
                    help='Name of source domain (e.g., "gadras", "sim_v1")')
parser.add_argument('--target-domain', '-t', required=True,
                    help='Name of target domain (e.g., "geant4", "exp_v1")')
parser.add_argument('--seed', '-r', type=int, default=0,
                    help='Random seed for reproducibility')
parser.add_argument('--train-time', '-T', type=float, default=120,
                    help='Training time in minutes')
parser.add_argument('--eval-metric', '-e', default='accuracy',
                    choices=['accuracy', 'ape', 'f1', 'cosine', 'crossentropy'],
                    help='Evaluation metric to use (in addition to crossentropy loss)')

args = parser.parse_args()

method = args.method
architecture = args.architecture
source_domain = args.source_domain
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
print(f"UDA Training Configuration")
print(f"{'='*60}")
print(f"Method:           {method}")
print(f"Source domain:    {source_domain}")
print(f"Target domain:    {target_domain}")
print(f"Architecture:     {architecture}")
print(f"Eval metric:      {eval_metric}")
print(f"Random seed:      {seed}")
print(f"Training time:    {args.train_time} minutes")
print(f"{'='*60}\n")

random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

if architecture == "MLP":
    pretrained_model = MLP()
elif architecture == "CNN":
    pretrained_model = CNN()
elif architecture == "BaselineTBNN":
    pretrained_model = BaselineTBNN()
elif architecture.startswith("TBNN"):
    pretrained_model = TBNN()
else:
    raise ValueError(f"Unknown architecture: {architecture}")
pretrained_model.load(os.path.join(OUT_DIR, f"Baseline/{source_domain}/{architecture}/model_{seed}.json"))
source_model = pretrained_model.model

source_training_ss = read_hdf(os.path.join(DATA_DIR, f"{source_domain}/ss_train.h5"))
source_validation_ss = read_hdf(os.path.join(DATA_DIR, f"{source_domain}/ss_val.h5"))
source_testing_ss = read_hdf(os.path.join(DATA_DIR, f"{source_domain}/ss_test.h5"))
target_training_ss = read_hdf(os.path.join(DATA_DIR, f"{target_domain}/ss_train.h5"))
target_validation_ss = read_hdf(os.path.join(DATA_DIR, f"{target_domain}/ss_val.h5"))
target_testing_ss = read_hdf(os.path.join(DATA_DIR, f"{target_domain}/ss_test.h5"))

hyperparams = load_best_uda_hyperparameters(method, architecture, source_domain, target_domain)

print(f"Loaded hyperparameters for {method} {architecture}:")
for key, value in hyperparams.items():
    print(f"    {key} = {value}")

learning_rate = hyperparams["learning_rate"]
weight_decay = hyperparams["weight_decay"]
batch_size = hyperparams["batch_size"]
dropout = hyperparams["dropout"]

metrics = [CategoricalCrossentropy()]

if method == "ADDA":
    dt_lr_ratio = hyperparams["dt_lr_ratio"]
    discriminator_hidden_layers = hyperparams["discriminator_hidden_layers"]

    d_optimizer = AdamW(learning_rate=learning_rate*np.sqrt(dt_lr_ratio), weight_decay=weight_decay)
    t_optimizer = AdamW(learning_rate=learning_rate/np.sqrt(dt_lr_ratio), weight_decay=weight_decay)
    model = ADDA(d_optimizer=d_optimizer, t_optimizer=t_optimizer, source_model=source_model,
                 discriminator_hidden_layers=discriminator_hidden_layers, dropout=dropout, metrics=metrics)

elif method == "DAN":
    lmbda = hyperparams["lmbda"]
    sigma = hyperparams["sigma"]
    kernel_num = hyperparams["kernel_num"]

    optimizer = AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    model = DAN(optimizer=optimizer, source_model=source_model, lmbda=lmbda,
                 sigma=sigma, kernel_num=kernel_num, dropout=dropout, metrics=metrics)

elif method == "DANN":
    df_lr_ratio = hyperparams["df_lr_ratio"]
    grl_hidden_layers = hyperparams["grl_hidden_layers"]
    use_da_scheduler = hyperparams["use_da_scheduler"]
    da_param = hyperparams["da_param"]

    d_optimizer = AdamW(learning_rate=learning_rate*np.sqrt(df_lr_ratio), weight_decay=weight_decay)
    f_optimizer = AdamW(learning_rate=learning_rate/np.sqrt(df_lr_ratio), weight_decay=weight_decay)
    model = DANN(d_optimizer=d_optimizer, f_optimizer=f_optimizer, source_model=source_model,
                 grl_hidden_layers=grl_hidden_layers, use_da_scheduler=use_da_scheduler,
                 da_param=da_param, dropout=dropout, metrics=metrics)
    
elif method == "DeepCORAL":
    lmbda = hyperparams["lmbda"]

    optimizer = AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    model = DeepCORAL(optimizer=optimizer, source_model=source_model, lmbda=lmbda,
                      dropout=dropout, metrics=metrics)

elif method == "DeepJDOT":
    ot_weight = hyperparams["ot_weight"]
    sinkhorn_reg = hyperparams["sinkhorn_reg"]
    num_sinkhorn_iters = hyperparams["num_sinkhorn_iters"]
    jdot_alpha = hyperparams["jdot_alpha"]
    jdot_beta = hyperparams["jdot_beta"]

    optimizer = AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    model = DeepJDOT(optimizer=optimizer, source_model=source_model, ot_weight=ot_weight,
                     sinkhorn_reg=sinkhorn_reg, num_sinkhorn_iters=num_sinkhorn_iters,
                     jdot_alpha=jdot_alpha, jdot_beta=jdot_beta, dropout=dropout, metrics=metrics)
    
elif method == "MeanTeacher":
    consistency_weight = hyperparams["consistency_weight"]
    ema_decay = hyperparams["ema_decay"]
    effective_counts = hyperparams["effective_counts"]

    optimizer = AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    model = MeanTeacher(optimizer=optimizer, source_model=source_model, consistency_weight=consistency_weight,
                        ema_decay=ema_decay, effective_counts=effective_counts, dropout=dropout, metrics=metrics)

elif method == "SimCLR":
    temperature = hyperparams["temperature"]
    contrastive_weight = hyperparams["contrastive_weight"]
    projection_dim = hyperparams["projection_dim"]
    effective_counts = hyperparams["effective_counts"]

    optimizer = AdamW(learning_rate=learning_rate, weight_decay=weight_decay)

    model = SimCLR(optimizer=optimizer, source_model=source_model, temperature=temperature, contrastive_weight=contrastive_weight,
                   projection_dim=projection_dim, effective_counts=effective_counts, dropout=dropout, metrics=metrics)

model.fit(source_training_ss, target_training_ss, source_validation_ss,
          target_validation_ss, batch_size=batch_size, patience=40,
          verbose=True, training_time=train_time, es_monitor="tgt_val_loss")

outdir = os.path.join(OUT_DIR, f"UDA/{method}/{source_domain}_to_{target_domain}/{architecture}")
os.makedirs(outdir, exist_ok=True)

model_filename = os.path.join(outdir, f"model_{seed}.json")
model.save(model_filename, overwrite=True)
print(f"Saved model to {model_filename}")

source_crossentropy = model.calc_crossentropy(source_testing_ss)
target_crossentropy = model.calc_crossentropy(target_testing_ss)
calc_metric = getattr(model, metric_methods[eval_metric])
source_eval_score = calc_metric(source_testing_ss)
target_eval_score = calc_metric(target_testing_ss)

print(f"For UDA {method} {architecture} model ({source_domain} → {target_domain}):")
print(f"    {source_domain}_crossentropy = {source_crossentropy:.4g}")
print(f"    {source_domain}_{eval_metric} = {source_eval_score:.4g}")
print(f"    {target_domain}_crossentropy = {target_crossentropy:.4g}")
print(f"    {target_domain}_{eval_metric} = {target_eval_score:.4g}")

output_filename = os.path.join(outdir, "models.csv")
file_exists = os.path.exists(output_filename)
with open(output_filename, "a", newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["seed", f"{source_domain}_crossentropy", f"{source_domain}_{eval_metric}",
                            f"{target_domain}_crossentropy", f"{target_domain}_{eval_metric}"])
    writer.writerow([seed, source_crossentropy, source_eval_score,
                        target_crossentropy, target_eval_score])
    print(f"Saved output to {output_filename}")
