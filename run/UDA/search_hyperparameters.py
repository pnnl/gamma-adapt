import numpy as np
import tensorflow as tf
import sys
import os
import argparse
import optuna
from optuna.samplers import TPESampler
from riid import read_hdf
from riid.models import MLP, CNN, TBNN, BaselineTBNN, ADDA, DANN, DAN, DeepCORAL, DeepJDOT, MeanTeacher, SimCLR
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.metrics import CategoricalCrossentropy
import random
import multiprocessing as mp
from multiprocessing import Queue
import time
from ..utils.hyperparameter_utils import convert_uda_params

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

def run_single_trial(method, source_domain, target_domain, architecture, train_time, converted_params, trial_number, eval_metric, result_queue):
    random.seed(trial_number)
    np.random.seed(trial_number)
    tf.random.set_seed(trial_number)
    
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
    source_model = pretrained_model.model
    
    source_training_ss = read_hdf(os.path.join(DATA_DIR, f"{source_domain}/ss_train.h5"))
    source_validation_ss = read_hdf(os.path.join(DATA_DIR, f"{source_domain}/ss_val.h5"))
    target_training_ss = read_hdf(os.path.join(DATA_DIR, f"{target_domain}/ss_train.h5"))
    target_validation_ss = read_hdf(os.path.join(DATA_DIR, f"{target_domain}/ss_val.h5"))
    
    learning_rate = converted_params["learning_rate"]
    weight_decay = converted_params["weight_decay"]
    batch_size = converted_params["batch_size"]
    dropout = converted_params["dropout"]
    
    metrics = [CategoricalCrossentropy()]

    if method == "ADDA":        
        discriminator_hidden_layers = converted_params["discriminator_hidden_layers"]
        dt_lr_ratio = converted_params["dt_lr_ratio"]
        d_optimizer = AdamW(learning_rate=learning_rate * np.sqrt(dt_lr_ratio), weight_decay=weight_decay)
        t_optimizer = AdamW(learning_rate=learning_rate / np.sqrt(dt_lr_ratio), weight_decay=weight_decay)

        model = ADDA(
            d_optimizer=d_optimizer,
            t_optimizer=t_optimizer,
            source_model=source_model,
            discriminator_hidden_layers=discriminator_hidden_layers,
            dropout=dropout,
            metrics=metrics
        )

    elif method == "DAN":
        lmbda = converted_params["lmbda"]
        sigma = converted_params["sigma"]
        kernel_num = converted_params["kernel_num"]
        optimizer = AdamW(learning_rate=learning_rate, weight_decay=weight_decay)

        model = DAN(
            optimizer=optimizer,
            source_model=source_model,
            lmbda=lmbda,
            sigma=sigma,
            kernel_num=kernel_num,
            dropout=dropout,
            metrics=metrics
        )

    elif method == "DANN":
        grl_hidden_layers = converted_params["grl_hidden_layers"]
        use_da_scheduler = converted_params["use_da_scheduler"]
        da_param = converted_params["da_param"]
        df_lr_ratio = converted_params["df_lr_ratio"]
        d_optimizer = AdamW(learning_rate=learning_rate * np.sqrt(df_lr_ratio), weight_decay=weight_decay)
        f_optimizer = AdamW(learning_rate=learning_rate / np.sqrt(df_lr_ratio), weight_decay=weight_decay)

        model = DANN(
            d_optimizer=d_optimizer,
            f_optimizer=f_optimizer,
            source_model=source_model,
            grl_hidden_layers=grl_hidden_layers,
            use_da_scheduler=use_da_scheduler,
            da_param=da_param,
            dropout=dropout,
            metrics=metrics
        )

    elif method == "DeepCORAL":
        lmbda = converted_params["lmbda"] 
        optimizer = AdamW(learning_rate=learning_rate, weight_decay=weight_decay)

        model = DeepCORAL(
            optimizer=optimizer,
            source_model=source_model,
            lmbda=lmbda,
            dropout=dropout,
            metrics=metrics
        )

    elif method == "DeepJDOT":
        ot_weight = converted_params["ot_weight"]
        sinkhorn_reg = converted_params["sinkhorn_reg"]
        num_sinkhorn_iters = converted_params["num_sinkhorn_iters"]
        jdot_alpha = converted_params["jdot_alpha"]
        jdot_beta = converted_params["jdot_beta"]
        optimizer = AdamW(learning_rate=learning_rate, weight_decay=weight_decay)

        model = DeepJDOT(
            optimizer=optimizer,
            source_model=source_model,
            ot_weight=ot_weight,
            sinkhorn_reg=sinkhorn_reg,
            num_sinkhorn_iters=num_sinkhorn_iters,
            jdot_alpha=jdot_alpha,
            jdot_beta=jdot_beta,
            dropout=dropout,
            metrics=metrics
        )

    elif method == "MeanTeacher":
        consistency_weight = converted_params["consistency_weight"]
        ema_decay = converted_params["ema_decay"]
        effective_counts = converted_params["effective_counts"]
        optimizer = AdamW(learning_rate=learning_rate, weight_decay=weight_decay)

        model = MeanTeacher(
            optimizer=optimizer,
            source_model=source_model,
            consistency_weight=consistency_weight,
            ema_decay=ema_decay,
            effective_counts=effective_counts,
            dropout=dropout,
            metrics=metrics
        )

    elif method == "SimCLR":
        temperature = converted_params["temperature"]
        contrastive_weight = converted_params["contrastive_weight"]
        projection_dim = converted_params["projection_dim"]
        effective_counts = converted_params["effective_counts"]
        optimizer = AdamW(learning_rate=learning_rate, weight_decay=weight_decay)

        model = SimCLR(
            optimizer=optimizer,
            source_model=source_model,
            temperature=temperature,
            contrastive_weight=contrastive_weight,
            projection_dim=projection_dim,
            effective_counts=effective_counts,
            dropout=dropout,
            metrics=metrics
        )

    history = model.fit(
        source_training_ss, target_training_ss,
        source_validation_ss, target_validation_ss,
        batch_size=batch_size,
        patience=20,
        verbose=False,
        training_time=train_time,
        es_monitor="tgt_val_loss"
    )

    target_val_crossentropy = float(model.calc_crossentropy(target_validation_ss))
    calc_metric = getattr(model, metric_methods[eval_metric])
    target_val_eval_score = float(calc_metric(target_validation_ss))
    
    results = {
        "target_val_crossentropy": target_val_crossentropy,
        "target_val_eval_score": target_val_eval_score,
        "eval_metric": eval_metric,
        "validation_checks": len(history["tgt_val_loss"]),
        "success": True
    }
    
    result_queue.put(results)

class OptunaNeuralNetOptimizer:
    
    def __init__(self, method, source_domain, target_domain, architecture, train_time, eval_metric):
        self.method = method
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.architecture = architecture
        self.train_time = train_time
        self.eval_metric = eval_metric
        self.study = self._create_study()
    
    def _create_study(self):
        storage_url = f"sqlite:///{os.path.join(OUT_DIR, f'UDA/{self.method}/{self.source_domain}_to_{self.target_domain}/{self.architecture}/optuna_study.db')}"
        study_name = f"{self.method}_{self.architecture}"
        
        os.makedirs(os.path.join(OUT_DIR, f"UDA/{self.method}/{self.source_domain}_to_{self.target_domain}/{self.architecture}"), exist_ok=True)
        
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
            "learning_rate": trial.suggest_float("learning_rate", 1e-7, 1e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True),
            "log2_bs": trial.suggest_int("log2_bs", 6, 9),
            "dropout": trial.suggest_float("dropout", 0.0, 0.4)
        }
        
        if self.method == "ADDA":
            params.update({
                "num_disc_layers": trial.suggest_int("num_disc_layers", 1, 2),
                "log2_base_disc": trial.suggest_int("log2_base_disc", 9, 12),
                "dt_lr_ratio": trial.suggest_float("dt_lr_ratio", 0.01, 100.0, log=True)
            })
            
        elif self.method == "DAN":
            params.update({
                "lmbda": trial.suggest_float("lmbda", 0.1, 1e4, log=True),
                "sigma": trial.suggest_float("sigma", 0.1, 1e2, log=True),
                "kernel_num_idx": trial.suggest_int("kernel_num_idx", 0, 6)
            })
            
        elif self.method == "DANN":
            params.update({
                "num_grl_layers": trial.suggest_int("num_grl_layers", 1, 2),
                "log2_base_grl": trial.suggest_int("log2_base_grl", 6, 11),
                "use_da_scheduler": trial.suggest_categorical("use_da_scheduler", [False]),
                "da_param": trial.suggest_float("da_param", 0.0, 1.0),
                "df_lr_ratio": trial.suggest_float("df_lr_ratio", 0.1, 10.0, log=True)
            })
            
        elif self.method == "DeepCORAL":
            params.update({
                "lmbda": trial.suggest_float("lmbda", 1e-1, 1e10, log=True)
            })
            
        elif self.method == "DeepJDOT":
            params.update({
                "ot_weight": trial.suggest_float("ot_weight", 1e-2, 1e2, log=True),
                "sinkhorn_reg": trial.suggest_float("sinkhorn_reg", 1e-2, 1e1, log=True),
                "num_sinkhorn_iters": trial.suggest_int("num_sinkhorn_iters", 5, 30),
                "jdot_alpha": trial.suggest_float("jdot_alpha", 1e-2, 1e1, log=True),
                "jdot_beta": trial.suggest_float("jdot_beta", 0.0, 2.0)
            })

        elif self.method == "MeanTeacher":
            params.update({
                "consistency_weight": trial.suggest_float("consistency_weight", 1e-1, 1e3, log=True),
                "ema_decay": trial.suggest_float("ema_decay", 0.95, 0.9999),
                "effective_counts": trial.suggest_float("effective_counts", 1e2, 1e5, log=True)
            })

        elif self.method == "SimCLR":
            params.update({
                "temperature": trial.suggest_float("temperature", 0.03, 0.5, log=True),
                "contrastive_weight": trial.suggest_float("contrastive_weight", 1e-2, 1e3, log=True),
                "num_proj_layers": trial.suggest_int("num_proj_layers", 1, 3),
                "log2_base_proj": trial.suggest_int("log2_base_proj", 6, 10),
                "effective_counts": trial.suggest_float("effective_counts", 1e2, 1e5, log=True)
            })
        
        return params
    
    def objective(self, trial):
        params = self.suggest_parameters(trial)
        converted_params = convert_uda_params(params, self.method)
        
        print(f"\n=== Trial {trial.number} ===")
        print(f"Method: {self.method}, Architecture: {self.architecture}")
        for key, value in converted_params.items():
            print(f"    {key} = {value}")
        
        result_queue = Queue()
        
        process = mp.Process(
            target=run_single_trial,
            args=(self.method, self.source_domain, self.target_domain, self.architecture, 
                  self.train_time, converted_params, trial.number, self.eval_metric, result_queue)
        )
        
        process.start()
        process.join()

        try:
            results = result_queue.get(timeout=60)
        except Exception as e:
            results = {"success": False, "error": f"Process timeout or no results: {str(e)}", "target_val_crossentropy": float('inf')}
        
        if results.get("success", False):
            trial.set_user_attr("validation_checks", results["validation_checks"])
            trial.set_user_attr(f"target_val_{results['eval_metric']}", results["target_val_eval_score"])
            
            print(f"Trial {trial.number} completed:")
            print(f"    target_val_crossentropy = {results['target_val_crossentropy']:.6f}")
            print(f"    target_val_{results['eval_metric']} = {results['target_val_eval_score']:.6f}")
        else:
            print(f"Trial {trial.number} failed: {results.get('error', 'Unknown error')}")
        
        return results["target_val_crossentropy"]
    
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
        best_params = convert_uda_params(best_trial.params, self.method)
        
        print(f"\nBest trial (#{best_trial.number}):")
        print(f"    target_val_crossentropy = {best_trial.value:.6f}")
        metric_key = f"target_val_{self.eval_metric}"
        metric_value = best_trial.user_attrs.get(metric_key, 'N/A')
        if metric_value != 'N/A':
            print(f"    {metric_key} = {metric_value:.6f}")
        else:
            print(f"    {metric_key} = {metric_value}")
        print(f"    validation_checks = {best_trial.user_attrs.get('validation_checks', 'N/A')}")
        
        print(f"\nBest hyperparameters:")
        for key, value in best_params.items():
            print(f"    {key} = {value}")

def main():
    parser = argparse.ArgumentParser(
        description='Search for optimal UDA hyperparameters',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--method', '-m', required=True,
                        choices=['ADDA', 'DAN', 'DANN', 'DeepCORAL', 'DeepJDOT', 'MeanTeacher', 'SimCLR'],
                        help='UDA method to use')
    parser.add_argument('--source-domain', '-s', required=True,
                        help='Name of source domain (e.g., "gadras", "sim_v1")')
    parser.add_argument('--target-domain', '-t', required=True,
                        help='Name of target domain (e.g., "geant4", "exp_v1")')
    parser.add_argument('--architecture', '-a', required=True,
                        choices=['MLP', 'CNN', 'BaselineTBNN', 'TBNN_linear_cls', 'TBNN_linear_gap',
                                 'TBNN_mlp_cls', 'TBNN_cnn_cls'],
                        help='Model architecture')
    parser.add_argument('--n-trials', '-n', type=int, default=100,
                        help='Number of Optuna trials to run')
    parser.add_argument('--train-time', '-T', type=float, default=15,
                        help='Training time per trial in minutes')
    parser.add_argument('--eval-metric', '-e', default='accuracy',
                        choices=['accuracy', 'ape', 'f1', 'cosine', 'crossentropy'],
                        help='Evaluation metric to use (in addition to crossentropy loss)')
    
    args = parser.parse_args()
    
    train_time = args.train_time * 60

    print(f"\n{'='*60}")
    print(f"UDA Hyperparameter Search")
    print(f"{'='*60}")
    print(f"Method:           {args.method}")
    print(f"Source domain:    {args.source_domain}")
    print(f"Target domain:    {args.target_domain}")
    print(f"Architecture:     {args.architecture}")
    print(f"Eval metric:      {args.eval_metric}")
    print(f"Number of trials: {args.n_trials}")
    print(f"Time per trial:   {args.train_time} minutes")
    print(f"{'='*60}\n")

    optimizer = OptunaNeuralNetOptimizer(
        args.method,
        args.source_domain,
        args.target_domain,
        args.architecture,
        train_time,
        args.eval_metric
    )
    study = optimizer.optimize(args.n_trials)
    
    optimizer.print_results()

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
