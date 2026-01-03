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
import random
import multiprocessing as mp
from multiprocessing import Queue
import time
from ..utils.hyperparameter_utils import convert_proxy_params, load_baseline_architecture_hyperparameters

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

def run_single_trial(architecture, source_domain, target_domain, train_time, converted_params, trial_number, eval_metric, result_queue):
    random.seed(trial_number)
    np.random.seed(trial_number)
    tf.random.set_seed(trial_number)

    training_ss = read_hdf(os.path.join(DATA_DIR, f"{source_domain}/ss_train.h5"))
    validation_ss = read_hdf(os.path.join(DATA_DIR, f"{source_domain}/ss_val.h5"))
    target_validation_ss = read_hdf(os.path.join(DATA_DIR, f"{target_domain}/ss_val.h5"))
    
    learning_rate = converted_params["learning_rate"]
    weight_decay = converted_params["weight_decay"]
    batch_size = converted_params["batch_size"]
    dropout = converted_params["dropout"]
    label_smoothing = converted_params["label_smoothing"]
    
    optimizer = AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    loss = CategoricalCrossentropyLoss(label_smoothing=label_smoothing)
    
    if architecture == "MLP":
        model = MLP(
            optimizer=optimizer,
            loss=loss,
            hidden_layers=converted_params["hidden_layers"],
            dropout=dropout,
            normalize="sqrt_zscore"
        )
    elif architecture == "CNN":
        model = CNN(
            optimizer=optimizer,
            loss=loss,
            filters=converted_params["filters"],
            kernel_sizes=converted_params["kernel_sizes"],
            dense_layer_sizes=converted_params["dense_layer_sizes"],
            dropout=dropout,
            normalize="sqrt_zscore"
        )
    elif architecture == "BaselineTBNN":
        model = BaselineTBNN(
            optimizer=optimizer,
            loss=loss,
            num_heads=converted_params["num_heads"],
            ff_dim=converted_params["ff_dim"],
            num_layers=converted_params["num_layers"],
            dropout=dropout,
            normalize="sqrt_zscore"
        )
    elif architecture.startswith("TBNN"):
        arch, embed_mode, readout = architecture.split("_")
        if embed_mode in ["mlp", "cnn"]:
            embed_inner = converted_params["embed_inner"]
        else:
            embed_inner = None
        model = TBNN(
            optimizer=optimizer,
            loss=loss,
            embed_mode=embed_mode,
            embed_inner=embed_inner,
            embed_dim=converted_params["embed_dim"],
            pos_encoding="learnable",
            num_heads=converted_params["num_heads"],
            ff_dim=converted_params["ff_dim"],
            num_layers=converted_params["num_layers"],
            patch_size=converted_params["patch_size"],
            stride=converted_params["stride"],
            readout=readout,
            dropout=dropout,
            normalize="sqrt_zscore"
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    history = model.fit(
        training_ss,
        validation_ss,
        batch_size=batch_size,
        patience=20,
        verbose=False,
        training_time=train_time,
        es_monitor="val_loss"
    )

    val_crossentropy = float(model.calc_crossentropy(validation_ss))
    calc_metric = getattr(model, metric_methods[eval_metric])
    val_eval_score = float(calc_metric(validation_ss))
    target_val_crossentropy = float(model.calc_crossentropy(target_validation_ss))
    target_val_eval_score = float(calc_metric(target_validation_ss))
    
    results = {
        "val_crossentropy": val_crossentropy,
        "val_eval_score": val_eval_score,
        "target_val_crossentropy": target_val_crossentropy,
        "target_val_eval_score": target_val_eval_score,
        "eval_metric": eval_metric,
        "epochs": len(history["loss"]),
        "success": True
    }

    result_queue.put(results)

class OptunaNeuralNetOptimizer:
    def __init__(self, architecture, source_domain, target_domain, train_time, use_arch_search, eval_metric):
        self.architecture = architecture
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.train_time = train_time
        self.use_arch_search = use_arch_search
        self.eval_metric = eval_metric
        if not use_arch_search:
            self.arch_params = load_baseline_architecture_hyperparameters(architecture, source_domain)
        self.study = self._create_study()

    def _create_study(self):
        storage_url = f"sqlite:///{os.path.join(OUT_DIR, f'Baseline/{self.source_domain}/{self.architecture}/optuna_study.db')}"
        study_name = f"{self.source_domain}_{self.architecture}"
        
        os.makedirs(os.path.join(OUT_DIR, f"Baseline/{self.source_domain}/{self.architecture}"), exist_ok=True)
        
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
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 2e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True),
            "log2_bs": trial.suggest_int("log2_bs", 5, 9),
            "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.0)
        }
        if self.architecture == "MLP" or self.architecture == "CNN":
            params["dropout"] = trial.suggest_float("dropout", 0.0, 0.4)
        elif self.architecture == "BaselineTBNN" or self.architecture.startswith("TBNN"):
            params["dropout"] = trial.suggest_float("dropout", 0.0, 0.1)

        if self.use_arch_search:
            if self.architecture == "MLP":
                params.update({
                    "num_layers": trial.suggest_int("num_layers", 1, 4),
                    "log2_base_dim": trial.suggest_int("log2_base_dim", 9, 13)
                })
                
            elif self.architecture == "CNN":
                params.update({
                    "num_conv_layers": trial.suggest_int("num_conv_layers", 1, 2),
                    "log2_base_filters": trial.suggest_int("log2_base_filters", 4, 7),
                    "kernel_idx": trial.suggest_int("kernel_idx", 0, 3),
                    "num_dense_layers": trial.suggest_int("num_dense_layers", 1, 2),
                    "log2_dense_base": trial.suggest_int("log2_dense_base", 9, 12)
                })
                
            elif self.architecture == "BaselineTBNN":
                params.update({
                    "log2_num_heads": trial.suggest_int("log2_num_heads", 0, 3),
                    "log2_ff_dim_factor": trial.suggest_int("log2_ff_dim_factor", 1, 8),
                    "num_layers": trial.suggest_int("num_layers", 1, 8)
                })

            elif self.architecture.startswith("TBNN"):
                arch, embed_mode, readout = self.architecture.split("_")
                
                params.update({
                    "log2_num_heads": trial.suggest_int("log2_num_heads", 0, 3),
                    "log2_ff_dim_factor": trial.suggest_int("log2_ff_dim_factor", 1, 4),
                    "num_layers": trial.suggest_int("num_layers", 1, 5),
                    "log2_patch_size": trial.suggest_int("log2_patch_size", 4, 6),
                    "log2_stride_factor": trial.suggest_int("log2_stride_factor", 0, 0)
                })

                if embed_mode == "linear":
                    params["log2_embed_dim"] = trial.suggest_int("log2_embed_dim", 3, 10)

                elif embed_mode == "mlp":
                    params["log2_embed_dim"] = trial.suggest_int("log2_embed_dim", 3, 10)
                    params["log2_embed_inner_factor"] = trial.suggest_int("log2_embed_inner_factor", 0, 1)
                    
                elif embed_mode == "cnn":
                    params["log2_embed_dim"] = trial.suggest_int("log2_embed_dim", 3, 10)
                    params["log2_embed_inner"] = trial.suggest_int("log2_embed_inner", 3, 7)

        return params

    def objective(self, trial):
        params = self.suggest_parameters(trial)
        
        if not self.use_arch_search:
            converted_training_params = {
                "learning_rate": params["learning_rate"],
                "weight_decay": params["weight_decay"],
                "batch_size": 2 ** int(params["log2_bs"]),
                "dropout": params["dropout"],
                "label_smoothing": params["label_smoothing"]
            }
            converted_params = {**self.arch_params, **converted_training_params}
        else:
            converted_params = convert_proxy_params(params, self.architecture)
        
        print(f"\n=== Trial {trial.number} ===")
        print(f"Architecture: {self.architecture}, Source: {self.source_domain}, Target: {self.target_domain}")
        for key, value in converted_params.items():
            print(f"    {key} = {value}")

        result_queue = Queue()

        process = mp.Process(
            target=run_single_trial,
            args=(self.architecture, self.source_domain, self.target_domain, self.train_time, converted_params, trial.number, self.eval_metric, result_queue)
        )

        process.start()
        process.join()

        results = result_queue.get(timeout=60)

        if results.get("success", False):
            trial.set_user_attr("epochs", results["epochs"])
            trial.set_user_attr("val_crossentropy", results["val_crossentropy"])
            trial.set_user_attr(f"val_{results['eval_metric']}", results["val_eval_score"])
            trial.set_user_attr("target_val_crossentropy", results["target_val_crossentropy"])
            trial.set_user_attr(f"target_val_{results['eval_metric']}", results["target_val_eval_score"])

            print(f"Trial {trial.number} completed:")
            print(f"    val_crossentropy = {results['val_crossentropy']:.6f}")
            print(f"    val_{results['eval_metric']} = {results['val_eval_score']:.6f}")
            print(f"    target_val_crossentropy = {results['target_val_crossentropy']:.6f}")
            print(f"    target_val_{results['eval_metric']} = {results['target_val_eval_score']:.6f}")
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

        if not self.use_arch_search:
            converted_training_params = {
                "learning_rate": best_trial.params["learning_rate"],
                "weight_decay": best_trial.params["weight_decay"],
                "batch_size": 2 ** int(best_trial.params["log2_bs"]),
                "dropout": best_trial.params["dropout"],
                "label_smoothing": best_trial.params["label_smoothing"]
            }
            best_params = {**self.arch_params, **converted_training_params}
        else:
            best_params = convert_proxy_params(best_trial.params, self.architecture)

        print(f"\nBest trial (#{best_trial.number}):")
        print(f"    val_crossentropy = {best_trial.value:.6f}")
        
        val_metric_key = f"val_{self.eval_metric}"
        val_metric_value = best_trial.user_attrs.get(val_metric_key, 'N/A')
        if val_metric_value != 'N/A':
            print(f"    {val_metric_key} = {val_metric_value:.6f}")
        else:
            print(f"    {val_metric_key} = {val_metric_value}")
        
        target_val_ce = best_trial.user_attrs.get('target_val_crossentropy', 'N/A')
        if target_val_ce != 'N/A':
            print(f"    target_val_crossentropy = {target_val_ce:.6f}")
        else:
            print(f"    target_val_crossentropy = {target_val_ce}")
        
        target_metric_key = f"target_val_{self.eval_metric}"
        target_metric_value = best_trial.user_attrs.get(target_metric_key, 'N/A')
        if target_metric_value != 'N/A':
            print(f"    {target_metric_key} = {target_metric_value:.6f}")
        else:
            print(f"    {target_metric_key} = {target_metric_value}")
        
        print(f"    epochs = {best_trial.user_attrs.get('epochs', 'N/A')}")
        
        print(f"\nBest hyperparameters:")
        for key, value in best_params.items():
            print(f"    {key} = {value}")

def main():
    parser = argparse.ArgumentParser(
        description='Search for optimal hyperparameters for baseline model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--source-domain', '-s', required=True,
                        help='Name of source domain (e.g., "gadras", "sim_v1")')
    parser.add_argument('--architecture', '-a', required=True,
                        choices=['MLP', 'CNN', 'BaselineTBNN', 'TBNN_linear_cls', 'TBNN_linear_gap',
                                 'TBNN_mlp_cls', 'TBNN_cnn_cls'],
                        help='Model architecture to optimize')
    parser.add_argument('--target-domain', '-t', default=None,
                        help='Name of target domain for evaluation (default: same as source)')
    parser.add_argument('--n-trials', '-n', type=int, default=100,
                        help='Number of Optuna trials to run')
    parser.add_argument('--train-time', '-T', type=float, default=15,
                        help='Training time per trial in minutes')
    parser.add_argument('--eval-metric', '-e', default='accuracy',
                        choices=['accuracy', 'ape', 'f1', 'cosine', 'crossentropy'],
                        help='Evaluation metric to use (in addition to crossentropy loss)')

    args = parser.parse_args()

    source_domain = args.source_domain
    architecture = args.architecture
    target_domain = args.target_domain if args.target_domain else source_domain
    n_trials = args.n_trials
    train_time = args.train_time * 60

    # Determine if we should search architecture hyperparameters
    # Search if this is the first domain, skip if we already have arch params from another domain
    use_arch_search = True
    try:
        load_baseline_architecture_hyperparameters(architecture, source_domain)
        use_arch_search = False
        print(f"Using existing architecture hyperparameters from {source_domain}")
    except:
        print(f"Searching for both architecture and training hyperparameters")

    print(f"\n{'='*60}")
    print(f"Baseline Hyperparameter Search Configuration")
    print(f"{'='*60}")
    print(f"Source domain:    {source_domain}")
    print(f"Target domain:    {target_domain}")
    print(f"Architecture:     {architecture}")
    print(f"Eval metric:      {args.eval_metric}")
    print(f"Number of trials: {n_trials}")
    print(f"Time per trial:   {args.train_time} minutes")
    print(f"{'='*60}\n")

    optimizer = OptunaNeuralNetOptimizer(architecture, source_domain, target_domain, train_time, use_arch_search, args.eval_metric)
    study = optimizer.optimize(n_trials)
    
    optimizer.print_results()

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
