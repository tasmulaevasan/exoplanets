import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

class TabDataset(Dataset):
    def __init__(self, X_num, X_cat, y):
        self.X_num = X_num.astype(np.float32)
        self.X_cat = X_cat.astype(np.int64) if X_cat is not None else None
        self.y = y.astype(np.int64)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        num = torch.from_numpy(self.X_num[idx])
        cat = torch.from_numpy(self.X_cat[idx]) if self.X_cat is not None else torch.empty(0, dtype=torch.long)
        label = torch.tensor(self.y[idx], dtype=torch.long)
        return num, cat, label

def load_datasets(datasets=None, custom_data_path=None):
    dataframes = []
    if custom_data_path and os.path.exists(custom_data_path):
        print(f"Loading custom data: {custom_data_path}")
        dataframes.append(pd.read_csv(custom_data_path))
    default_path = os.path.join(SCRIPT_DIR, "data", "exoplanet_dataset.csv")
    if not dataframes:
        if os.path.exists(default_path):
            print(f"Loading prepared dataset: {default_path}")
            dataframes.append(pd.read_csv(default_path))
        else:
            error_msg = (
                f"\n❌ ERROR: Dataset file not found!\n"
                f"Expected: {default_path}\n"
                f"\nPlease ensure exoplanet_dataset.csv exists in the data directory.\n"
                f"Run prepare_dataset.py to create it from raw NASA files.\n"
            )
            raise RuntimeError(error_msg)
    df = pd.concat(dataframes, ignore_index=True)
    if "label" not in df.columns:
        raise RuntimeError(
            f"\n❌ ERROR: 'label' column not found in dataset!\n"
            f"The dataset must have a 'label' column with values 0 (False Positive) or 1 (Confirmed).\n"
            f"Run prepare_dataset.py to process raw NASA files.\n"
        )
    return df

def prepare_data(df, num_cols, cat_cols, test_size=0.15, seed=42):
    df_labeled = df[df["label"].isin([0, 1])].reset_index(drop=True)
    df_candidates = df[df["label"] == -1].reset_index(drop=True)
    print(f"\nData:")
    print(f"  Labeled: {len(df_labeled)} (CONFIRMED: {(df_labeled['label']==1).sum()}, FP: {(df_labeled['label']==0).sum()})")
    print(f"  CANDIDATE: {len(df_candidates)}")
    for c in num_cols[:]:
        if c not in df_labeled.columns:
            print(f"⚠️ Missing column: {c}")
            num_cols.remove(c)
    X_num_labeled = df_labeled[num_cols].fillna(0.).values
    X_num_candidates = df_candidates[num_cols].fillna(0.).values if len(df_candidates) > 0 else None
    label_encoders = {}
    X_cat_labeled = np.zeros((len(df_labeled), len(cat_cols)), dtype=np.int64)
    cat_cardinalities = []
    for i, c in enumerate(cat_cols):
        if c not in df_labeled.columns:
            continue
        le = LabelEncoder()
        df_labeled[c] = df_labeled[c].fillna("NA").astype(str)
        X_cat_labeled[:, i] = le.fit_transform(df_labeled[c])
        label_encoders[c] = le
        cat_cardinalities.append(len(le.classes_))
    if len(df_candidates) > 0:
        X_cat_candidates = np.zeros((len(df_candidates), len(cat_cols)), dtype=np.int64)
        for i, c in enumerate(cat_cols):
            if c in df_candidates.columns:
                df_candidates[c] = df_candidates[c].fillna("NA").astype(str)
                X_cat_candidates[:, i] = label_encoders[c].transform(
                    df_candidates[c].map(lambda x: x if x in label_encoders[c].classes_ else "NA")
                )
    else:
        X_cat_candidates = None
    y_labeled = df_labeled["label"].astype(int).values
    Xn_train, Xn_val, Xc_train, Xc_val, y_train, y_val = train_test_split(
        X_num_labeled, X_cat_labeled, y_labeled,
        test_size=test_size, random_state=seed, stratify=y_labeled
    )
    scaler = RobustScaler()
    Xn_train = scaler.fit_transform(Xn_train)
    Xn_val = scaler.transform(Xn_val)
    if X_num_candidates is not None:
        Xn_candidates = scaler.transform(X_num_candidates)
    return {
        'train': (Xn_train, Xc_train, y_train),
        'val': (Xn_val, Xc_val, y_val),
        'candidates': (Xn_candidates, X_cat_candidates) if X_num_candidates is not None else None,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'cat_cardinalities': cat_cardinalities,
        'num_cols': num_cols,
        'cat_cols': cat_cols
    }

def create_dataloaders(data_dict, batch_size=32):
    Xn_train, Xc_train, y_train = data_dict['train']
    Xn_val, Xc_val, y_val = data_dict['val']
    train_ds = TabDataset(Xn_train, Xc_train, y_train)
    val_ds = TabDataset(Xn_val, Xc_val, y_val)
    class_counts = Counter(y_train)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[y] for y in y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    candidate_loader = None
    if data_dict['candidates'] is not None:
        Xn_candidates, Xc_candidates = data_dict['candidates']
        candidate_ds = TabDataset(Xn_candidates, Xc_candidates, np.zeros(len(Xn_candidates)))
        candidate_loader = DataLoader(candidate_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, candidate_loader