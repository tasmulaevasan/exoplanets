import os
import torch
import numpy as np
from torch import nn
from model import TabTransformerModel
from utils import load_checkpoint
from config import get_model_config, MODEL_CONFIGS

DEVICE = "cpu"

_model_cache = {}
def get_default_model_type():
    pretrained_path = MODEL_CONFIGS["pretrained"]["model_file"]
    if os.path.exists(pretrained_path):
        return "pretrained"
    return "user"

def load_model(model_type=None):
    global _model_cache
    if model_type is None:
        model_type = get_default_model_type()
    if model_type in _model_cache:
        return _model_cache[model_type]
    config = get_model_config(model_type)
    model_path = config["model_file"]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    hp = checkpoint.get('hyperparameters', config["hyperparameters"])
    model = TabTransformerModel(
        num_features=checkpoint.get('num_features', len(checkpoint['num_cols'])),
        cat_cardinalities=checkpoint['cat_cardinalities'],
        cat_embed_dim=hp.get('cat_embed_dim', config["hyperparameters"]["cat_embed_dim"]),
        d_model=hp.get('d_model', config["hyperparameters"]["d_model"]),
        n_heads=hp.get('n_heads', config["hyperparameters"]["n_heads"]),
        n_layers=hp.get('n_layers', config["hyperparameters"]["n_layers"]),
        mlp_hidden=hp.get('mlp_hidden', config["hyperparameters"]["mlp_hidden"]),
        out_dim=1,
        dropout=hp.get('dropout', config["hyperparameters"]["dropout"])
    )
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    _model_cache[model_type] = {
        'model': model,
        'scaler': checkpoint['scaler'],
        'label_encoders': checkpoint['label_encoders'],
        'num_cols': checkpoint['num_cols'],
        'cat_cols': checkpoint['cat_cols'],
        'model_type': model_type
    }
    return _model_cache[model_type]

def predict_single(input_features: dict, model_type=None):
    cache = load_model(model_type)
    model = cache['model']
    scaler = cache['scaler']
    label_encoders = cache['label_encoders']
    num_cols = cache['num_cols']
    cat_cols = cache['cat_cols']
    X_num = np.array([[float(input_features.get(c, 0.0)) for c in num_cols]], dtype=np.float32)
    X_num = scaler.transform(X_num)
    X_num_t = torch.from_numpy(X_num)
    X_cat_list = []
    for c in cat_cols:
        val = input_features.get(c, None)
        if val is None or val not in label_encoders[c].classes_:
            val = label_encoders[c].classes_[0]
        encoded = label_encoders[c].transform([val])[0]
        X_cat_list.append(encoded)
    X_cat = np.array([X_cat_list], dtype=np.int64)
    X_cat_t = torch.from_numpy(X_cat)
    with torch.no_grad():
        logits = model(X_num_t, X_cat_t)
        prob = torch.sigmoid(logits).item()
    return {
        'prediction': "CONFIRMED" if prob >= 0.5 else "FALSE POSITIVE",
        'confidence': float(prob),
        'classification': 1 if prob >= 0.5 else 0
    }

def predict_batch(df, model_type=None):
    cache = load_model(model_type)
    model = cache['model']
    scaler = cache['scaler']
    label_encoders = cache['label_encoders']
    num_cols = cache['num_cols']
    cat_cols = cache['cat_cols']
    missing_num = [c for c in num_cols if c not in df.columns]
    if missing_num:
        for c in missing_num:
            df[c] = 0.0
    X_num = df[num_cols].fillna(0.).values.astype(np.float32)
    X_num = scaler.transform(X_num)
    X_num_t = torch.from_numpy(X_num)
    X_cat = np.zeros((len(df), len(cat_cols)), dtype=np.int64)
    for i, c in enumerate(cat_cols):
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)
            encoded_vals = []
            for val in df[c]:
                if val and val in label_encoders[c].classes_:
                    encoded_vals.append(label_encoders[c].transform([val])[0])
                else:
                    encoded_vals.append(label_encoders[c].transform([label_encoders[c].classes_[0]])[0])
            X_cat[:, i] = np.array(encoded_vals)
        else:
            X_cat[:, i] = label_encoders[c].transform([label_encoders[c].classes_[0]])[0]
    X_cat_t = torch.from_numpy(X_cat)
    with torch.no_grad():
        logits = model(X_num_t, X_cat_t)
        probs = torch.sigmoid(logits).cpu().numpy()
    predictions = []
    for i, prob in enumerate(probs):
        row = df.iloc[i]
        pred = {
            "id": row.get("kepid", row.get("kepoi_name", row.get("tic_id", f"OBJ-{i:05d}"))),
            "prediction": "Exoplanet" if prob >= 0.5 else "False Positive",
            "confidence": float(prob),
            "period": f"{row.get('period', 0):.2f} days" if 'period' in row else "N/A",
            "radius": f"{row.get('radius', 0):.2f} R⊕" if 'radius' in row else "N/A"
        }
        predictions.append(pred)
    return predictions