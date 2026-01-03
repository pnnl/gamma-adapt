import numpy as np
import tensorflow as tf
import sys
import os
import argparse
import optuna
from optuna.samplers import TPESampler
from riid import read_hdf
from riid.models import MLP, CNN, TBNN, BaselineTBNN
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.losses import CategoricalCrossentropy as CategoricalCrossentropyLoss
from tensorflow.keras.layers import Dropout
import random
import multiprocessing as mp
from multiprocessing import Queue
import time
from ..utils.hyperparameter_utils import convert_sda_finetune_params, convert_sda_fromscratch_params, load_best_baseline_hyperparameters

# Get configurable paths from environment variables
DATA_DIR = os.environ.get('DATA_DIR', 'data')
OUT_DIR = os.environ.get('OUT_DIR', 'out')

# Map metric names to PyRIID method names
metric_methods = {
    'accuracy': 'calc_accuracy',
    'ape': 'calc_APE_score',
    'f1': 'calc_f1_score',
    'cosine': 'calc_cosine_similarity',
    'crossentropy': 'calc_crossentropy'
}

def run_finetune_trial(source_domain, target_domain, architecture, training_size, train_time, converted_params, trial_number, eval_metric, result_queue):
    random.seed(trial_number)
    np.random.seed(trial_number)
    tf.random.set_seed(trial_number)
    
    # Load data
    if training_size is None:
        target_training_ss = read_hdf(os.path.join(DATA_DIR, f"{target_domain}/ss_train.h5"))
    else:
        target_training_ss = read_hdf(os.path.join(DATA_DIR, f"{target_domain}/ss_train.h5"), row_slice=(0, training_size))
    target_validation_ss = read_hdf(os.path.join(DATA_DIR, f"{target_domain}/ss_val.h5"))

    learning_rate = converted_params["learning_rate"]
    weight_decay = converted_params["weight_decay"]
    batch_size = converted_params["batch_size"]
    dropout = converted_params["dropout"]
    label_smoothing = converted_params["label_smoothing"]
    frozen_indices = converted_params["frozen_indices"]

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
    pretrained_model.load(os.path.join(OUT_DIR, f"Baseline/{source_domain}/{architecture}/model_0.json"))

    for layer in pretrained_model.model.layers:
        if isinstance(layer, Dropout):
            layer.rate = dropout

    for idx in frozen_indices:
        pretrained_model.model.layers[idx].trainable = False

    optimizer = AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    loss = CategoricalCrossentropyLoss(label_smoothing=label_smoothing)
    pretrained_model.model.compile(
        optimizer=optimizer,
        loss=loss
    )

    history = pretrained_model.fit(
        target_training_ss,
        target_validation_ss,
        batch_size=batch_size,
        patience=5,
        verbose=False,
        training_time=train_time,
        es_monitor="val_loss"
    )

    val_crossentropy = float(pretrained_model.calc_crossentropy(target_validation_ss))
    calc_metric = getattr(pretrained_model, metric_methods[eval_metric])
    val_eval_score = float(calc_metric(target_validation_ss))

    results = {
        "val_crossentropy": val_crossentropy,
        "val_eval_score": val_eval_score,
        "eval_metric": eval_metric,
        "epochs": len(history["loss"]),
        "success": True
    }
    
    result_queue.put(results)

def run_fromscratch_trial(source_domain, target_domain, architecture, training_size, train_time, converted_params, trial_number, eval_metric, result_queue):
    random.seed(trial_number)
    np.random.seed(trial_number)
    tf.random.set_seed(trial_number)
    
    # Load data
    if training_size is None:
        target_training_ss = read_hdf(os.path.join(DATA_DIR, f"{target_domain}/ss_train.h5"))
    else:
        target_training_ss = read_hdf(os.path.join(DATA_DIR, f"{target_domain}/ss_train.h5"), row_slice=(0, training_size))
    target_validation_ss = read_hdf(os.path.join(DATA_DIR, f"{target_domain}/ss_val.h5"))

    learning_rate = converted_params["learning_rate"]
    weight_decay = converted_params["weight_decay"]
    batch_size = converted_params["batch_size"]
    dropout = converted_params["dropout"]
    label_smoothing = converted_params["label_smoothing"]
    optimizer = AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    loss = CategoricalCrossentropyLoss(label_smoothing=label_smoothing)

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

    history = model.fit(
        target_training_ss,
        target_validation_ss,
        batch_size=batch_size,
        patience=5,
        verbose=False,
        training_time=train_time,
        es_monitor="val_categorical_crossentropy"
    )

    val_crossentropy = float(model.calc_crossentropy(target_validation_ss))
    calc_metric = getattr(model, metric_methods[eval_metric])
    val_eval_score = float(calc_metric(target_validation_ss))

    results = {
        "val_crossentropy": val_crossentropy,
        "val_eval_score": val_eval_score,
        "eval_metric": eval_metric,
        "epochs": len(history["loss"]),
        "success": True
    }
    
    result_queue.put(results)

class OptunaNeuralNetOptimizer:
    
    def __init__(self, source_domain, target_domain, architecture, mode, training_size, train_time, eval_metric):
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.architecture = architecture
        self.mode = mode
        self.training_size = training_size
        self.train_time = train_time
        self.eval_metric = eval_metric
        
        # Determine training_size_id for study naming
        if training_size is None:
            self.training_size_id = "full"
            actual_size = 1000000  # Assume large dataset for batch size calculation
        else:
            self.training_size_id = training_size
            actual_size = training_size
        
        # Calculate batch size range
        batch_size_min = 32
        batch_size_max = max(32, min(actual_size, 512))
        self.log2_bs_min = int(np.log2(batch_size_min))
        self.log2_bs_max = int(np.log2(batch_size_max))
        
        if mode in ["finetune", "finetune_start", "finetune_end"]:
            with tf.device('/CPU:0'):
                if architecture == "MLP":
                    temp_model = MLP()
                elif architecture == "CNN":
                    temp_model = CNN()
                elif architecture == "BaselineTBNN":
                    temp_model = BaselineTBNN()
                elif architecture.startswith("TBNN"):
                    temp_model = TBNN()
                else:
                    raise ValueError(f"Unknown architecture: {architecture}")
                temp_model.load(os.path.join(OUT_DIR, f"Baseline/{source_domain}/{architecture}/model_0.json"))
                self.trainable_layer_indices = [idx for idx, layer in enumerate(temp_model.model.layers) if layer.trainable_weights]
        
        self.study = self._create_study()
    
    def _create_study(self):
        if self.mode in ["finetune", "finetune_start", "finetune_end"]:
            storage_url = f"sqlite:///{os.path.join(OUT_DIR, f'SDA/finetune/{self.source_domain}_to_{self.target_domain}/{self.architecture}/optuna_study.db')}"
            os.makedirs(os.path.join(OUT_DIR, f"SDA/finetune/{self.source_domain}_to_{self.target_domain}/{self.architecture}"), exist_ok=True)
        else:
            storage_url = f"sqlite:///{os.path.join(OUT_DIR, f'SDA/{self.mode}/{self.source_domain}_to_{self.target_domain}/{self.architecture}/optuna_study.db')}"
            os.makedirs(os.path.join(OUT_DIR, f"SDA/{self.mode}/{self.source_domain}_to_{self.target_domain}/{self.architecture}"), exist_ok=True)
        
        study_name = f"{self.mode}_{self.architecture}_size_{self.training_size_id}"
        
        seed = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
        max_retries = 10
        for attempt in range(max_retries):
            try:
                return optuna.create_study(
                    study_name=study_name,
                    storage=storage_url,
                    load_if_exists=True,
                    direction="minimize",
                    sampler=TPESampler(
                        seed=seed,
                        multivariate=True,
                        warn_independent_sampling=False
                    )
                )
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Database creation conflict (attempt {attempt + 1}/{max_retries}), retrying...")
                    time.sleep(random.uniform(0.1, 1.0))
                else:
                    raise e
    
    def suggest_parameters(self, trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True),
            "log2_bs": trial.suggest_int("log2_bs", self.log2_bs_min, self.log2_bs_max),
            "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.0)
        }
        if self.architecture == "MLP" or self.architecture == "CNN":
            params["dropout"] = trial.suggest_float("dropout", 0.0, 0.5)
        elif self.architecture == "BaselineTBNN" or self.architecture.startswith("TBNN"):
            params["dropout"] = trial.suggest_float("dropout", 0.0, 0.2)
        
        if self.mode in ["finetune_start", "finetune_end"]:
            params.update({
                "num_layers_to_freeze": trial.suggest_int("num_layers_to_freeze", 0, max(0, len(self.trainable_layer_indices) - 1))
            })
        
        return params
    
    def objective(self, trial):
        params = self.suggest_parameters(trial)
        
        if self.mode in ["finetune", "finetune_start", "finetune_end"]:
            converted_params = convert_sda_finetune_params(params, self.trainable_layer_indices, self.mode)
            trial_func = run_finetune_trial
        else:
            converted_params = convert_sda_fromscratch_params(params)
            trial_func = run_fromscratch_trial
        
        print(f"\n=== Trial {trial.number} ===")
        print(f"Architecture: {self.architecture}, Mode: {self.mode}")
        for key, value in converted_params.items():
            print(f"    {key} = {value}")
        
        result_queue = Queue()
        
        process = mp.Process(
            target=trial_func,
            args=(self.source_domain, self.target_domain, self.architecture, self.training_size, 
                  self.train_time, converted_params, trial.number, self.eval_metric, result_queue)
        )
        
        process.start()
        process.join()

        try:
            results = result_queue.get(timeout=60)
        except Exception as e:
            results = {"success": False, "error": f"Process timeout or no results: {str(e)}", "val_crossentropy": float('inf')}
        
        if results.get("success", False):
            trial.set_user_attr("epochs", results["epochs"])
            trial.set_user_attr(f"val_{results['eval_metric']}", results["val_eval_score"])
            
            print(f"Trial {trial.number} completed:")
            print(f"    val_crossentropy = {results['val_crossentropy']:.6f}")
            print(f"    val_{results['eval_metric']} = {results['val_eval_score']:.6f}")
        else:
            print(f"Trial {trial.number} failed: {results.get('error', 'Unknown error')}")
        
        return results["val_crossentropy"]
    
    def optimize(self, n_trials):
        self.study.optimize(self.objective, n_trials=n_trials)
        return self.study
    
    def print_results(self):
        print(f"\n{'='*60}")
        print(f"OPTIMIZATION COMPLETE")
        print(f"{'='*60}")
        
        print(f"Study statistics:")
        print(f"    Total trials: {len(self.study.trials)}")
        print(f"    Completed trials: {len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
        print(f"    Failed trials: {len([t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL])}")
        
        best_trial = self.study.best_trial
        
        if self.mode in ["finetune", "finetune_start", "finetune_end"]:
            best_params = convert_sda_finetune_params(best_trial.params, self.trainable_layer_indices, self.mode)
        else:
            best_params = convert_sda_fromscratch_params(best_trial.params)
        
        print(f"\nBest trial (#{best_trial.number}):")
        print(f"    val_crossentropy = {best_trial.value:.6f}")
        metric_key = f"val_{self.eval_metric}"
        metric_value = best_trial.user_attrs.get(metric_key, 'N/A')
        if metric_value != 'N/A':
            print(f"    {metric_key} = {metric_value:.6f}")
        else:
            print(f"    {metric_key} = {metric_value}")
        print(f"    epochs = {best_trial.user_attrs.get('epochs', 'N/A')}")
        
        print(f"\nBest hyperparameters:")
        for key, value in best_params.items():
            print(f"    {key} = {value}")

def main():
    parser = argparse.ArgumentParser(
        description='Search for optimal SDA hyperparameters',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--mode', '-m', default='finetune',
                        choices=['fromscratch', 'finetune', 'finetune_start', 'finetune_end'],
                        help='Training mode to search hyperparameters for (default: finetune)')
    parser.add_argument('--architecture', '-a', required=True,
                        choices=['MLP', 'CNN', 'BaselineTBNN', 'TBNN_linear_cls', 'TBNN_linear_gap',
                                 'TBNN_mlp_cls', 'TBNN_cnn_cls'],
                        help='Model architecture')
    parser.add_argument('--source-domain', '-s', required=True,
                        help='Name of source domain (e.g., "gadras", "sim_v1")')
    parser.add_argument('--target-domain', '-t', required=True,
                        help='Name of target domain (e.g., "geant4", "exp_v1")')
    parser.add_argument('--training-size', '-z', type=int, default=None,
                        help='Number of training samples to use (default: use entire dataset)')
    parser.add_argument('--n-trials', '-n', type=int, default=100,
                        help='Number of Optuna trials to run')
    parser.add_argument('--train-time', '-T', type=float, default=15,
                        help='Training time per trial in minutes')
    parser.add_argument('--eval-metric', '-e', default='accuracy',
                        choices=['accuracy', 'ape', 'f1', 'cosine', 'crossentropy'],
                        help='Evaluation metric to use (in addition to crossentropy loss)')
    
    args = parser.parse_args()
    
    train_time = args.train_time * 60
    
    size_label = "full" if args.training_size is None else str(args.training_size)
    
    print(f"\n{'='*60}")
    print(f"SDA Hyperparameter Search")
    print(f"{'='*60}")
    print(f"Mode:             {args.mode}")
    print(f"Source domain:    {args.source_domain}")
    print(f"Target domain:    {args.target_domain}")
    print(f"Architecture:     {args.architecture}")
    print(f"Training size:    {size_label}")
    print(f"Eval metric:      {args.eval_metric}")
    print(f"Number of trials: {args.n_trials}")
    print(f"Time per trial:   {args.train_time} minutes")
    print(f"{'='*60}\n")
    
    optimizer = OptunaNeuralNetOptimizer(
        args.source_domain,
        args.target_domain,
        args.architecture,
        args.mode,
        args.training_size,
        train_time,
        args.eval_metric
    )
    study = optimizer.optimize(args.n_trials)
    optimizer.print_results()

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
