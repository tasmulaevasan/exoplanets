import random
import numpy as np
import torch

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def pseudo_label_candidates(model, candidate_loader, threshold, device):
    model.eval()
    pseudo_data = []
    with torch.no_grad():
        for xb_num, xb_cat, _ in candidate_loader:
            xb_num = xb_num.to(device)
            xb_cat = xb_cat.to(device)
            logits = model(xb_num, xb_cat)
            probs = torch.sigmoid(logits).cpu().numpy()
            for i, prob in enumerate(probs):
                if prob >= threshold:
                    pseudo_data.append((xb_num[i].cpu().numpy(), xb_cat[i].cpu().numpy(), 1))
                elif prob <= (1 - threshold):
                    pseudo_data.append((xb_num[i].cpu().numpy(), xb_cat[i].cpu().numpy(), 0))
    return pseudo_data

def calculate_metrics(model, dataloader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb_num, xb_cat, yb in dataloader:
            xb_num = xb_num.to(device)
            xb_cat = xb_cat.to(device)
            logits = model(xb_num, xb_cat)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds.extend(probs.tolist())
            trues.extend(yb.numpy().tolist())
    return {'predictions': preds, 'labels': trues}

def save_checkpoint(model, scaler, label_encoders, metadata, path):
    checkpoint = {
        'model_state': model.state_dict(),
        'scaler': scaler,
        'label_encoders': label_encoders,
        **metadata
    }
    torch.save(checkpoint, path)

def load_checkpoint(path, device='cpu'):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    return checkpoint