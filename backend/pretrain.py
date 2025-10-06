"""
Standalone pre-training script for large TabTransformer model
Does NOT import train.py - completely independent
"""

import os
import sys
import json
import torch
from datetime import datetime
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix, classification_report
from model import TabTransformerModel, FocalLoss
from dataset import load_datasets, prepare_data, create_dataloaders
from utils import seed_everything, pseudo_label_candidates, calculate_metrics, save_checkpoint
from config import MODEL_CONFIGS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def check_environment():
    data_dir = os.path.join(SCRIPT_DIR, "data")
    if not os.path.exists(data_dir):
        print("=" * 80)
        print("⚠️  WARNING: Data directory not found!")
        print("=" * 80)
        print(f"Expected data at: {data_dir}")
        response = input("\nContinue anyway? (yes/no): ")
        if response.lower() != 'yes':
            sys.exit(1)

def pretrain():
    seed_everything(42)
    check_environment()
    print("=" * 80)
    print("PRE-TRAINING LARGE MODEL")
    print("=" * 80)
    config = MODEL_CONFIGS["pretrained"]
    hp = config["hyperparameters"]
    print(f"\nModel: {config['name']}")
    print(f"\nHyperparameters:")
    print(f"  Device: {hp['device']}")
    print(f"  Epochs: {hp['epochs']}")
    print(f"  Batch Size: {hp['batch_size']}")
    print(f"  Learning Rate: {hp['learning_rate']}")
    print(f"  Architecture: {hp['cat_embed_dim']}/{hp['d_model']}/{hp['n_heads']}/{hp['n_layers']}/{hp['mlp_hidden']}")
    print(f"  Pseudo-label start: Epoch {hp['pseudo_label_start_epoch']}")
    device = hp['device']
    if device == 'cuda' and not torch.cuda.is_available():
        print("\nERROR: CUDA requested but not available!")
        print("Options:")
        print("  1. Run on a machine with NVIDIA GPU and CUDA")
        print("  2. Edit config.py to set device='cpu' (WARNING: 10-20x slower)")
        response = input("\nContinue with CPU? (yes/no): ")
        if response.lower() != 'yes':
            sys.exit(1)
        device = 'cpu'
    print(f"\nUsing device: {device}")
    print(f"Training will use: data/exoplanet_dataset.csv")
    response = input("\nStart pre-training? (yes/no): ")
    if response.lower() != 'yes':
        print("Cancelled.")
        sys.exit(0)
    print("\n" + "=" * 80)
    print("LOADING DATA...")
    print("=" * 80)
    num_cols = [
        "period", "radius", "depth", "star_temp", "star_radius", "insolation",
        "log_period", "log_insolation", "radius_star_ratio", "density_proxy"
    ]
    cat_cols = ["source"]
    df = load_datasets()
    data_dict = prepare_data(df, num_cols, cat_cols, test_size=0.15, seed=42)
    train_loader, val_loader, candidate_loader = create_dataloaders(data_dict, batch_size=hp['batch_size'])
    print("\n" + "=" * 80)
    print("CREATING MODEL...")
    print("=" * 80)
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp['learning_rate'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp['epochs'])
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    print("\n" + "=" * 80)
    print("STARTING TRAINING...")
    print("=" * 80)
    best_auc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_auc": [], "val_acc": [], "val_f1": []}
    for epoch in range(1, hp['epochs'] + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        num_batches = len(train_loader)
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
            if batch_idx % 10 == 0:
                running_acc = running_correct / running_total if running_total > 0 else 0.0
                print(f"[EPOCH {epoch}/{hp['epochs']}] Batch {batch_idx}/{num_batches} | Loss: {loss.item():.4f} | Acc: {running_acc:.4f}")
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
        print(f"\n[EPOCH {epoch}/{hp['epochs']}] COMPLETE")
        print(f"  Loss: {avg_loss:.4f} | AUC: {auc:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}\n")
        if epoch >= hp['pseudo_label_start_epoch'] and candidate_loader is not None and epoch % 5 == 0:
            pseudo_data = pseudo_label_candidates(
                model, candidate_loader, hp['pseudo_label_threshold'], device
            )
            if len(pseudo_data) > 0:
                print(f"  Added {len(pseudo_data)} pseudo-labels")
        if auc > best_auc:
            best_auc = auc
            patience_counter = 0
            os.makedirs(os.path.dirname(config['model_file']), exist_ok=True)
            save_checkpoint(
                model=model,
                scaler=data_dict['scaler'],
                label_encoders=data_dict['label_encoders'],
                metadata={
                    'num_cols': data_dict['num_cols'],
                    'cat_cols': data_dict['cat_cols'],
                    'cat_cardinalities': data_dict['cat_cardinalities'],
                    'timestamp': datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
                    'hyperparameters': hp
                },
                path=config['model_file']
            )
            print(f"  Saved best model (AUC={best_auc:.4f}) -> {config['model_file']}")
        else:
            patience_counter += 1
            if patience_counter >= hp['patience']:
                print(f"  Early stopping after {epoch} epochs")
                break

    print("\n" + "=" * 80)
    print("FINAL EVALUATION...")
    print("=" * 80)
    checkpoint = torch.load(config['model_file'], map_location=device, weights_only=False)
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
        "history": history,
        "timestamp": checkpoint['timestamp'],
        "hyperparameters": hp,
        "model_filename": os.path.basename(config['model_file'])
    }
    with open(config['metrics_file'], 'w') as f:
        json.dump(final_metrics, f, indent=2)
    print(f"\nAccuracy: {final_metrics['accuracy']:.4f}")
    print(f"AUC-ROC:  {final_metrics['auc']:.4f}")
    print(f"F1-Score: {final_metrics['f1_score']:.4f}")
    print("\nClassification Report:")
    print(classification_report(trues, preds_bin, target_names=["FALSE POSITIVE", "CONFIRMED"]))
    print("\n" + "=" * 80)
    print("PRE-TRAINING COMPLETED")
    print("=" * 80)
    print(f"\nModel saved to: {config['model_file']}")
    print(f"Metrics saved to: {config['metrics_file']}")

if __name__ == "__main__":
    try:
        pretrain()
    except Exception as e:
        print("\n" + "=" * 80)
        print("PRE-TRAINING FAILED!")
        print("=" * 80)
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)