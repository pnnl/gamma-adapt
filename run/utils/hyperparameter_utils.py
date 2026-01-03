import optuna
import tensorflow as tf
import os
from riid.models import MLP, CNN, TBNN, BaselineTBNN

# Get configurable paths from environment variables
OUT_DIR = os.environ.get('OUT_DIR', 'out')

def make_convolutional_layers(num_conv_layers, log2_base_filters, kernel_size):
    """Create convolutional layer configuration for CNN architecture."""
    base_filters = 2 ** int(log2_base_filters)
    filters = []
    kernel_sizes = []
    for i in range(int(num_conv_layers)):
        filters.append(int(base_filters * (2 ** i)))
        kernel_sizes.append(kernel_size)
    return filters, kernel_sizes

def make_dense_layers(num_dense_layers, log2_base_dim):
    """Create dense layer sizes for MLP and CNN architectures."""
    base_dense_dim = 2 ** int(log2_base_dim)
    dense_layer_sizes = []
    for i in range(int(num_dense_layers)):
        size = int(base_dense_dim // (2 ** i))
        dense_layer_sizes.append(size)
    return tuple(dense_layer_sizes)

def convert_proxy_params(params, architecture):
    """Convert optuna proxy parameters to actual model hyperparameters."""
    converted = {}
    
    converted["learning_rate"] = params["learning_rate"]
    converted["weight_decay"] = params["weight_decay"]
    converted["batch_size"] = 2 ** int(params["log2_bs"])
    converted["dropout"] = params["dropout"]
    converted["label_smoothing"] = params["label_smoothing"]
    
    if architecture == "MLP":
        num_layers = int(params["num_layers"])
        log2_base_dim = int(params["log2_base_dim"])
        hidden_layers = make_dense_layers(num_layers, log2_base_dim)
        converted["hidden_layers"] = hidden_layers
        
    elif architecture == "CNN":
        num_conv_layers = int(params["num_conv_layers"])
        log2_base_filters = int(params["log2_base_filters"])
        kernel_idx = int(params["kernel_idx"])
        kernel_size = [3, 5, 7, 9][kernel_idx]
        filters, kernel_sizes = make_convolutional_layers(num_conv_layers, log2_base_filters, kernel_size)
        
        num_dense_layers = int(params["num_dense_layers"])
        log2_dense_base = int(params["log2_dense_base"])
        dense_layer_sizes = make_dense_layers(num_dense_layers, log2_dense_base)
        
        converted["filters"] = filters
        converted["kernel_sizes"] = kernel_sizes
        converted["dense_layer_sizes"] = dense_layer_sizes

    elif architecture == "BaselineTBNN":
        num_heads = 2 ** int(params["log2_num_heads"])
        ff_dim_factor = 2 ** int(params["log2_ff_dim_factor"])
        ff_dim = 32 * ff_dim_factor
        num_layers = int(params["num_layers"])
        
        converted.update({
            "num_heads": num_heads,
            "ff_dim": ff_dim,
            "num_layers": num_layers
        })

    elif architecture.startswith("TBNN"):
        arch, embed_mode, readout = architecture.split("_")
        
        num_heads = 2 ** int(params["log2_num_heads"])
        ff_dim_factor = 2 ** int(params["log2_ff_dim_factor"])
        
        num_layers = int(params["num_layers"])
        patch_size = 2 ** int(params["log2_patch_size"])
        stride_factor = 2 ** int(params["log2_stride_factor"])
        stride = patch_size // stride_factor
        
        converted.update({
            "num_heads": num_heads,
            "num_layers": num_layers,
            "patch_size": patch_size,
            "stride": stride
        })

        if embed_mode == "linear":
            embed_dim = 2 ** int(params["log2_embed_dim"])
            converted["embed_dim"] = embed_dim
            converted["ff_dim"] = embed_dim * ff_dim_factor
        
        elif embed_mode == "mlp":
            embed_dim = 2 ** int(params["log2_embed_dim"])
            embed_inner_factor = 2 ** int(params["log2_embed_inner_factor"])
            converted["embed_dim"] = embed_dim
            converted["embed_inner"] = embed_dim * embed_inner_factor
            converted["ff_dim"] = embed_dim * ff_dim_factor

        elif embed_mode == "cnn":
            embed_dim = 2 ** int(params["log2_embed_dim"])
            embed_inner = 2 ** int(params["log2_embed_inner"])
            converted["embed_dim"] = embed_dim
            converted["embed_inner"] = embed_inner
            converted["ff_dim"] = embed_dim * ff_dim_factor
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    return converted

def convert_sda_finetune_params(params, trainable_layer_indices, mode):
    """Convert SDA finetune parameters."""
    converted = {}
    
    converted["learning_rate"] = params["learning_rate"]
    converted["weight_decay"] = params["weight_decay"]
    converted["batch_size"] = 2 ** int(params["log2_bs"])
    converted["dropout"] = params["dropout"]
    converted["label_smoothing"] = params["label_smoothing"]
    
    if mode == "finetune":
        converted["frozen_indices"] = []
    else:
        num_layers_to_freeze = params["num_layers_to_freeze"]
        
        if mode == "finetune_start":
            trainable_indices_to_freeze = list(range(num_layers_to_freeze))
        elif mode == "finetune_end":
            start_idx = len(trainable_layer_indices) - num_layers_to_freeze
            trainable_indices_to_freeze = list(range(start_idx, len(trainable_layer_indices)))
        
        frozen_indices = [trainable_layer_indices[i] for i in trainable_indices_to_freeze]
        converted["frozen_indices"] = frozen_indices
    
    return converted

def convert_sda_fromscratch_params(params):
    """Convert SDA fromscratch parameters."""
    converted = {}
    
    converted["learning_rate"] = params["learning_rate"]
    converted["weight_decay"] = params["weight_decay"]
    converted["batch_size"] = 2 ** int(params["log2_bs"])
    converted["dropout"] = params["dropout"]
    converted["label_smoothing"] = params["label_smoothing"]
    
    return converted

def convert_uda_params(params, method):
    """Convert UDA method-specific proxy parameters to actual hyperparameters."""
    converted = {}
    
    converted["learning_rate"] = params["learning_rate"]
    converted["weight_decay"] = params["weight_decay"]
    converted["batch_size"] = 2 ** int(params["log2_bs"])
    converted["dropout"] = params["dropout"]
    
    if method == "ADDA":
        num_disc_layers = int(params["num_disc_layers"])
        log2_base_disc = int(params["log2_base_disc"])
        discriminator_hidden_layers = make_dense_layers(num_disc_layers, log2_base_disc)
        
        converted["discriminator_hidden_layers"] = discriminator_hidden_layers
        converted["dt_lr_ratio"] = params["dt_lr_ratio"]
        
    elif method == "DAN":
        converted["lmbda"] = params["lmbda"]
        converted["sigma"] = params["sigma"]
        kernel_num_idx = int(params["kernel_num_idx"])
        converted["kernel_num"] = 3 + 2 * kernel_num_idx
        
    elif method == "DANN":
        num_grl_layers = int(params["num_grl_layers"])
        log2_base_grl = int(params["log2_base_grl"])
        grl_hidden_layers = make_dense_layers(num_grl_layers, log2_base_grl)
        
        converted["grl_hidden_layers"] = grl_hidden_layers
        converted["use_da_scheduler"] = params["use_da_scheduler"]
        converted["da_param"] = params["da_param"]
        converted["df_lr_ratio"] = params["df_lr_ratio"]
        
    elif method == "DeepCORAL":
        converted["lmbda"] = params["lmbda"]
        
    elif method == "DeepJDOT":
        converted["ot_weight"] = params["ot_weight"]
        converted["sinkhorn_reg"] = params["sinkhorn_reg"]
        converted["num_sinkhorn_iters"] = int(params["num_sinkhorn_iters"])
        converted["jdot_alpha"] = params["jdot_alpha"]
        converted["jdot_beta"] = params["jdot_beta"]

    elif method == "MeanTeacher":
        converted["consistency_weight"] = params["consistency_weight"]
        converted["ema_decay"] = params["ema_decay"]
        converted["effective_counts"] = params["effective_counts"]

    elif method == "SimCLR":
        num_proj_layers = int(params["num_proj_layers"])
        log2_base_proj = int(params["log2_base_proj"])
        projection_dim = make_dense_layers(num_proj_layers, log2_base_proj)

        converted["temperature"] = params["temperature"]
        converted["contrastive_weight"] = params["contrastive_weight"]
        converted["projection_dim"] = projection_dim
        converted["effective_counts"] = params["effective_counts"]
    
    return converted

def load_baseline_architecture_hyperparameters(architecture, source_domain):
    """Load architecture hyperparameters from the source domain baseline study."""
    storage_url = f"sqlite:///{os.path.join(OUT_DIR, f'Baseline/{source_domain}/{architecture}/optuna_study.db')}"
    study_name = f"{source_domain}_{architecture}"
    
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    best_trial = study.best_trial
    converted_params = convert_proxy_params(best_trial.params, architecture)
    
    arch_params = {}
    
    if architecture == "MLP":
        arch_params["hidden_layers"] = converted_params["hidden_layers"]
    elif architecture == "CNN":
        arch_params["filters"] = converted_params["filters"]
        arch_params["kernel_sizes"] = converted_params["kernel_sizes"]
        arch_params["dense_layer_sizes"] = converted_params["dense_layer_sizes"]
    elif architecture == "BaselineTBNN":
        arch_params.update({
            "num_heads": converted_params["num_heads"],
            "ff_dim": converted_params["ff_dim"],
            "num_layers": converted_params["num_layers"]
        })
    elif architecture.startswith("TBNN"):
        arch, embed_mode, readout = architecture.split("_")
        arch_params.update({
            "embed_dim": converted_params["embed_dim"],
            "num_heads": converted_params["num_heads"],
            "ff_dim": converted_params["ff_dim"],
            "num_layers": converted_params["num_layers"],
            "patch_size": converted_params["patch_size"],
            "stride": converted_params["stride"]
        })
        if embed_mode in ["mlp", "cnn"]:
            arch_params["embed_inner"] = converted_params["embed_inner"]
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    return arch_params

def load_best_baseline_hyperparameters(architecture, source_domain):
    """Load the best hyperparameters from a Baseline optuna study."""
    arch_params = load_baseline_architecture_hyperparameters(architecture, source_domain)

    storage_url = f"sqlite:///{os.path.join(OUT_DIR, f'Baseline/{source_domain}/{architecture}/optuna_study.db')}"
    study_name = f"{source_domain}_{architecture}"

    study = optuna.load_study(study_name=study_name, storage=storage_url)
    best_trial = study.best_trial

    training_params = {
        "learning_rate": best_trial.params["learning_rate"],
        "weight_decay": best_trial.params["weight_decay"],
        "batch_size": 2 ** int(best_trial.params["log2_bs"]),
        "dropout": best_trial.params["dropout"],
        "label_smoothing": best_trial.params["label_smoothing"]
    }

    hyperparameters = {**arch_params, **training_params}
    
    if "epochs" in best_trial.user_attrs:
        hyperparameters["epochs"] = best_trial.user_attrs["epochs"]
    
    return hyperparameters

def load_best_sda_finetune_hyperparameters(architecture, source_domain, target_domain, training_size_id):
    storage_url = f"sqlite:///{os.path.join(OUT_DIR, f'SDA/finetune/{source_domain}_to_{target_domain}/{architecture}/optuna_study.db')}"
    finetune_study_name = f"finetune_{architecture}_size_{training_size_id}"
    start_study_name = f"finetune_start_{architecture}_size_{training_size_id}"
    end_study_name = f"finetune_end_{architecture}_size_{training_size_id}"
    
    # Try to load all three studies
    finetune_study = None
    start_study = None
    end_study = None
    
    try:
        finetune_study = optuna.load_study(study_name=finetune_study_name, storage=storage_url)
    except KeyError:
        pass
    
    try:
        start_study = optuna.load_study(study_name=start_study_name, storage=storage_url)
    except KeyError:
        pass
    
    try:
        end_study = optuna.load_study(study_name=end_study_name, storage=storage_url)
    except KeyError:
        pass
    
    # If no studies exist, try fallback to "full" or raise error
    if finetune_study is None and start_study is None and end_study is None:
        if training_size_id != "full":
            print(f"Warning: No hyperparameters found for training_size_id={training_size_id}, falling back to 'full'")
            return load_best_sda_finetune_hyperparameters(architecture, source_domain, target_domain, "full")
        else:
            raise KeyError(f"No SDA finetune hyperparameters found for {architecture} ({source_domain} → {target_domain}). Run hyperparameter search first.")
    
    # Select the best study among all available
    studies = []
    if finetune_study is not None:
        studies.append(("finetune", finetune_study))
    if start_study is not None:
        studies.append(("finetune_start", start_study))
    if end_study is not None:
        studies.append(("finetune_end", end_study))
    
    # Pick the study with the best validation loss
    selected_mode, selected_study = min(studies, key=lambda x: x[1].best_value)
    
    if len(studies) > 1:
        val_losses = ", ".join([f"{mode}: {study.best_value:.6f}" for mode, study in studies])
        print(f"Selected {selected_mode} mode (val_losses: {val_losses})")
    else:
        print(f"Using {selected_mode} mode (val_loss: {selected_study.best_value:.6f})")
    
    best_trial = selected_study.best_trial

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
        trainable_layer_indices = [idx for idx, layer in enumerate(temp_model.model.layers) if layer.trainable_weights]
    
    hyperparameters = convert_sda_finetune_params(best_trial.params, trainable_layer_indices, selected_mode)
    
    if "epochs" in best_trial.user_attrs:
        hyperparameters["epochs"] = best_trial.user_attrs["epochs"]
    
    return hyperparameters

def load_best_sda_fromscratch_hyperparameters(architecture, source_domain, target_domain, training_size_id):
    """Load the best hyperparameters from an SDA fromscratch optuna study."""
    storage_url = f"sqlite:///{os.path.join(OUT_DIR, f'SDA/fromscratch/{source_domain}_to_{target_domain}/{architecture}/optuna_study.db')}"
    study_name = f"fromscratch_{architecture}_size_{training_size_id}"
    
    # Try to load study for the specified training_size_id, fall back to "full" if not found
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
    except KeyError:
        if training_size_id != "full":
            print(f"Warning: No hyperparameters found for training_size_id={training_size_id}, falling back to 'full'")
            return load_best_sda_fromscratch_hyperparameters(architecture, source_domain, target_domain, "full")
        else:
            raise
    best_trial = study.best_trial

    hyperparameters = convert_sda_fromscratch_params(best_trial.params)
    
    if "epochs" in best_trial.user_attrs:
        hyperparameters["epochs"] = best_trial.user_attrs["epochs"]
    
    return hyperparameters

def load_best_uda_hyperparameters(method, architecture, source_domain, target_domain):
    """Load the best hyperparameters from a UDA optuna study."""
    storage_url = f"sqlite:///{os.path.join(OUT_DIR, f'UDA/{method}/{source_domain}_to_{target_domain}/{architecture}/optuna_study.db')}"
    study_name = f"{method}_{architecture}"
    
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    best_trial = study.best_trial
    
    hyperparameters = convert_uda_params(best_trial.params, method)
    
    if "epochs" in best_trial.user_attrs:
        hyperparameters["epochs"] = best_trial.user_attrs["epochs"]
    
    return hyperparameters
