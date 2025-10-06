import os
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_CONFIGS = {
    "pretrained": {
        "name": "Pre-trained Large Model",
        "description": "High-accuracy model trained on CUDA with extensive hyperparameters. Pre-trained offline using pretrain.py script.",
        "model_file": os.path.join(SCRIPT_DIR, "models", "tabtransformer_pretrained.pth"),
        "metrics_file": os.path.join(SCRIPT_DIR, "models", "metrics_pretrained.json"),
        "hyperparameters": {
            "cat_embed_dim": 64,
            "d_model": 128,
            "n_heads": 8,
            "n_layers": 4,
            "mlp_hidden": 256,
            "dropout": 0.15,
            "batch_size": 64,
            "epochs": 60,
            "learning_rate": 0.0003,
            "pseudo_label_start_epoch": 50,
            "pseudo_label_threshold": 0.95,
            "patience": 12,
            "device": "cuda"
        },
        "available": False
    },
    "user": {
        "name": "User Custom Model",
        "description": "Fast-training model optimized for CPU with reduced parameters",
        "model_file": os.path.join(SCRIPT_DIR, "models", "tabtransformer_user.pth"),
        "metrics_file": os.path.join(SCRIPT_DIR, "models", "metrics_user.json"),
        "hyperparameters": {
            "cat_embed_dim": 32,
            "d_model": 64,
            "n_heads": 4,
            "n_layers": 2,
            "mlp_hidden": 128,
            "dropout": 0.15,
            "batch_size": 32,
            "epochs": 30,
            "learning_rate": 0.0003,
            "pseudo_label_start_epoch": 20,
            "pseudo_label_threshold": 0.95,
            "patience": 8,
            "device": "cpu"
        },
        "available": True
    }
}

def get_model_config(model_type: str):
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model type: {model_type}. Must be 'pretrained' or 'user'")
    return MODEL_CONFIGS[model_type]

def get_available_models():
    models_info = []
    for model_type, config in MODEL_CONFIGS.items():
        model_exists = os.path.exists(config["model_file"])
        metrics_exists = os.path.exists(config["metrics_file"])
        metrics = None
        if metrics_exists:
            try:
                with open(config["metrics_file"], 'r') as f:
                    metrics = json.load(f)
            except:
                pass

        models_info.append({
            "type": model_type,
            "name": config["name"],
            "description": config["description"],
            "hyperparameters": config["hyperparameters"],
            "available": model_exists,
            "can_train": config["available"],
            "training_time_estimate": config["training_time_estimate"],
            "metrics": metrics
        })
    return models_info