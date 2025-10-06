import os
import json
import torch
from datetime import datetime
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from model import TabTransformerModel, FocalLoss
from dataset import load_datasets, prepare_data, create_dataloaders
from utils import seed_everything, pseudo_label_candidates, calculate_metrics, save_checkpoint
from config import get_model_config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NUM_COLS = [
    "period", "radius", "depth", "star_temp", "star_radius", "insolation",
    "log_period", "log_insolation", "radius_star_ratio", "density_proxy"
]
CAT_COLS = ["source"]
def train_model(
    learning_rate=None,
    epochs=None,
    batch_size=None,
    datasets=None,
    custom_data_path=None,
    progress_callback=None,
    model_output_path=None,
    metrics_output_path=None,
    model_type="user",
    **kwargs
):
    seed_everything(42)
    config = get_model_config(model_type)
    hp = config["hyperparameters"]
    learning_rate = learning_rate if learning_rate is not None else hp["learning_rate"]
    epochs = epochs if epochs is not None else hp["epochs"]
    batch_size = batch_size if batch_size is not None else hp["batch_size"]
    model_output_path = model_output_path or config["model_file"]
    metrics_output_path = metrics_output_path or config["metrics_file"]
    device = "cpu"
    print(f"\n{'='*80}")
    print(f"Training {config['name']}")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}, Batch: {batch_size}, LR: {learning_rate}")
    print(f"Architecture: {hp['cat_embed_dim']}/{hp['d_model']}/{hp['n_heads']}/{hp['n_layers']}/{hp['mlp_hidden']}")
    df = load_datasets(datasets, custom_data_path)
    data_dict = prepare_data(df, NUM_COLS, CAT_COLS, test_size=0.15, seed=42)
    train_loader, val_loader, candidate_loader = create_dataloaders(data_dict, batch_size=batch_size)
    model = TabTransformerModel(
        num_features=len(data_dict['num_cols']),
        cat_cardinalities=data_dict['cat_cardinalities'],
        cat_embed_dim=hp['cat_embed_dim'],
        d_model=hp['d_model'],
        n_heads=hp['n_heads'],
        n_layers=hp['n_layers'],
        mlp_hidden=hp['mlp_hidden'],
        dropout=hp['dropout']
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    best_auc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_auc": [], "val_acc": [], "val_f1": []}
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        num_batches = len(train_loader)
        print(f"[EPOCH {epoch}] Starting with {num_batches} batches...")
        for batch_idx, (xb_num, xb_cat, yb) in enumerate(train_loader):
            xb_num = xb_num.to(device)
            xb_cat = xb_cat.to(device)
            yb = yb.to(device).float()
            logits = model(xb_num, xb_cat)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item() * xb_num.size(0)
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                predictions = (probs >= 0.5).float()
                running_correct += (predictions == yb).sum().item()
                running_total += yb.size(0)
            if progress_callback is not None and batch_idx % 5 == 0:
                batch_progress = (epoch - 1 + batch_idx / num_batches) / epochs
                running_acc = running_correct / running_total if running_total > 0 else 0.0
                try:
                    progress_callback(
                        epoch,
                        epochs,
                        {
                            "loss": loss.item(),
                            "accuracy": running_acc,
                            "batch": batch_idx,
                            "total_batches": num_batches,
                            "progress_pct": int(batch_progress * 100)
                        }
                    )
                except Exception as e:
                    print(f"[CALLBACK ERROR] {e}")
        avg_loss = running_loss / len(train_loader.dataset)
        scheduler.step()
        val_metrics = calculate_metrics(model, val_loader, device)
        preds = val_metrics['predictions']
        trues = val_metrics['labels']
        auc = roc_auc_score(trues, preds)
        preds_bin = [1 if p >= 0.5 else 0 for p in preds]
        acc = accuracy_score(trues, preds_bin)
        f1 = f1_score(trues, preds_bin)
        history["train_loss"].append(avg_loss)
        history["val_auc"].append(auc)
        history["val_acc"].append(acc)
        history["val_f1"].append(f1)
        print(f"[EPOCH {epoch}/{epochs}] Loss: {avg_loss:.4f} | AUC: {auc:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")
        if progress_callback is not None:
            try:
                progress_callback(epoch, epochs, {
                    "loss": avg_loss,
                    "auc": auc,
                    "accuracy": acc,
                    "f1": f1
                })
            except Exception as e:
                print(f"[CALLBACK ERROR] {e}")
        if epoch >= hp['pseudo_label_start_epoch'] and candidate_loader is not None and epoch % 5 == 0:
            pseudo_data = pseudo_label_candidates(
                model, candidate_loader, hp['pseudo_label_threshold'], device
            )
            if len(pseudo_data) > 0:
                print(f"  → Added {len(pseudo_data)} pseudo-labels")
        if auc > best_auc:
            best_auc = auc
            patience_counter = 0
            os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
            save_checkpoint(
                model=model,
                scaler=data_dict['scaler'],
                label_encoders=data_dict['label_encoders'],
                metadata={
                    'num_cols': data_dict['num_cols'],
                    'cat_cols': data_dict['cat_cols'],
                    'cat_cardinalities': data_dict['cat_cardinalities'],
                    'timestamp': datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
                    'hyperparameters': {
                        'cat_embed_dim': hp['cat_embed_dim'],
                        'd_model': hp['d_model'],
                        'n_heads': hp['n_heads'],
                        'n_layers': hp['n_layers'],
                        'mlp_hidden': hp['mlp_hidden'],
                        'dropout': hp['dropout'],
                        'learning_rate': learning_rate,
                        'batch_size': batch_size,
                        'epochs': epochs
                    }
                },
                path=model_output_path
            )
            print(f"  Saved best model (AUC={best_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= hp['patience']:
                print(f"  Early stopping after {epoch} epochs")
                break
    checkpoint = torch.load(model_output_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    val_metrics = calculate_metrics(model, val_loader, device)
    preds = val_metrics['predictions']
    trues = val_metrics['labels']
    preds_bin = [1 if p >= 0.5 else 0 for p in preds]
    final_metrics = {
        "accuracy": float(accuracy_score(trues, preds_bin)),
        "auc": float(roc_auc_score(trues, preds)),
        "f1_score": float(f1_score(trues, preds_bin)),
        "confusion_matrix": confusion_matrix(trues, preds_bin).tolist(),
        "num_features": len(data_dict['num_cols']),
        "cat_features": len(data_dict['cat_cols']),
        "model_params": sum(p.numel() for p in model.parameters()),
        "training_samples": len(data_dict['train'][2]),
        "validation_samples": len(data_dict['val'][2]),
        "history": history,
        "timestamp": checkpoint['timestamp'],
        "model_filename": os.path.basename(model_output_path),
        "hyperparameters": {
            'cat_embed_dim': hp['cat_embed_dim'],
            'd_model': hp['d_model'],
            'n_heads': hp['n_heads'],
            'n_layers': hp['n_layers'],
            'mlp_hidden': hp['mlp_hidden'],
            'dropout': hp['dropout'],
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'epochs': epochs
        }
    }
    with open(metrics_output_path, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    print(f"\nFinal Results:")
    print(f"  Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"  AUC-ROC:  {final_metrics['auc']:.4f}")
    print(f"  F1-Score: {final_metrics['f1_score']:.4f}")
    print(f"\nraining complete! Model saved to {model_output_path}")
    return final_metrics

def train():
    return train_model()

if __name__ == "__main__":
    train()