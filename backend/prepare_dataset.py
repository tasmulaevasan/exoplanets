import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path('data')
OUTPUT = DATA / 'exoplanet_dataset.csv'

def load(file_path, dataset_name):
    df = pd.read_csv(file_path, low_memory=False)
    return df

def remove(df, columns, n_std=5):
    df_clean = df.copy()
    for col in columns:
        if col in df_clean.columns:
            mean = df_clean[col].mean()
            std = df_clean[col].std()
            df_clean = df_clean[
                (df_clean[col] >= mean - n_std * std) & 
                (df_clean[col] <= mean + n_std * std)
            ]
    return df_clean

def map_label(val):
    if pd.isna(val):
        return None
    val = str(val).strip().upper()
    if val in ['CONFIRMED', 'KP']:
        return 1
    if val in ['FALSE POSITIVE', 'FP']:
        return 0
    if val in ['CANDIDATE', 'PC', 'CP']:
        return -1
    return None

def process_k2(df):
    required_cols = ['pl_orbper', 'pl_rade', 'pl_trandep', 'st_teff', 'st_rad', 'pl_insol', 'disposition']
    df_clean = df.dropna(subset=required_cols).copy()
    df_clean = df_clean[[
        'pl_name', 'pl_orbper', 'pl_rade', 'pl_trandep', 
        'st_teff', 'st_rad', 'pl_insol', 'disposition'
    ]].rename(columns={
        'pl_orbper': 'period',
        'pl_rade': 'radius',
        'pl_trandep': 'depth',
        'st_teff': 'star_temp',
        'st_rad': 'star_radius',
        'pl_insol': 'insolation',
        'disposition': 'label_raw'
    })
    df_clean['source'] = 'K2'
    df_clean['label'] = df_clean['label_raw'].apply(map_label)
    return df_clean

def process_kepler(df):
    required_cols = ['koi_period', 'koi_prad', 'koi_depth', 'koi_steff', 'koi_srad', 'koi_insol', 'koi_disposition']
    df_clean = df.dropna(subset=required_cols).copy()
    df_clean = df_clean[[
        'kepoi_name', 'koi_period', 'koi_prad', 'koi_depth',
        'koi_steff', 'koi_srad', 'koi_insol', 'koi_disposition'
    ]].rename(columns={
        'kepoi_name': 'pl_name',
        'koi_period': 'period',
        'koi_prad': 'radius',
        'koi_depth': 'depth',
        'koi_steff': 'star_temp',
        'koi_srad': 'star_radius',
        'koi_insol': 'insolation',
        'koi_disposition': 'label_raw'
    })
    df_clean['source'] = 'Kepler'
    df_clean['label'] = df_clean['label_raw'].apply(map_label)
    return df_clean

def process_tess(df):
    required_cols = ['pl_orbper', 'pl_rade', 'pl_trandep', 'st_teff', 'st_rad', 'pl_insol', 'tfopwg_disp']
    df_clean = df.dropna(subset=required_cols).copy()
    df_clean = df_clean[[
        'toi', 'pl_orbper', 'pl_rade', 'pl_trandep',
        'st_teff', 'st_rad', 'pl_insol', 'tfopwg_disp'
    ]].rename(columns={
        'toi': 'pl_name',
        'pl_orbper': 'period',
        'pl_rade': 'radius',
        'pl_trandep': 'depth',
        'st_teff': 'star_temp',
        'st_rad': 'star_radius',
        'pl_insol': 'insolation',
        'tfopwg_disp': 'label_raw'
    })
    df_clean['source'] = 'TESS'
    df_clean['label'] = df_clean['label_raw'].apply(map_label)
    return df_clean

def engineered(df):
    df['log_period'] = np.log1p(df['period'])
    df['log_insolation'] = np.log1p(df['insolation'])
    df['radius_star_ratio'] = df['radius'] / df['star_radius']
    df['density_proxy'] = df['radius'] / (df['period'] ** (2/3))
    df['planet_temp_estimate'] = df['star_temp'] * np.sqrt(df['star_radius'] / (2 * df['period'] * 0.1))
    return df

def main():
    k2 = load(DATA / 'k2_planets_and_candidates.csv', 'K2')
    kepler = load(DATA / 'kepler_objects_of_interest.csv', 'Kepler')
    tess = load(DATA / 'tess_objects_of_interest.csv', 'TESS')
    k2_clean = process_k2(k2)
    kepler_clean = process_kepler(kepler)
    tess_clean = process_tess(tess)
    dataset = pd.concat([k2_clean, kepler_clean, tess_clean], ignore_index=True)
    dataset = dataset.dropna(subset=['label'])
    dataset = dataset[dataset['label'].isin([0, 1, -1])]
    dataset = dataset.drop_duplicates(subset=['pl_name'], keep='first')
    dataset = engineered(dataset)
    numeric_cols = ['period', 'radius', 'depth', 'star_temp', 'star_radius', 'insolation']
    dataset = remove(dataset, numeric_cols, n_std=5)
    label_counts = dataset['label'].value_counts().sort_index()
    for label, count in label_counts.items():
        label_int = int(label)
        label_name = {1: "CONFIRMED", 0: "FALSE POSITIVE", -1: "CANDIDATE"}[label_int]
        print(f"  {label_name:20s} ({label_int:2d}): {int(count):5d} ({count/len(dataset)*100:.1f}%)")
    for source, count in dataset['source'].value_counts().items():
        print(f"  {source:10s}: {count:5d}")
    print(dataset[numeric_cols].describe())
    nan_counts = dataset.isnull().sum()
    if nan_counts.sum() > 0:
        print(nan_counts[nan_counts > 0])
    dataset.to_csv(OUTPUT, index=False)
    confirmed_fp = dataset[dataset['label'].isin([0, 1])]
    candidates = dataset[dataset['label'] == -1]
    confirmed_fp.to_csv(DATA / 'labeled_data.csv', index=False)
    candidates.to_csv(DATA / 'candidates_for_pseudolabel.csv', index=False)

if __name__ == "__main__":
    main()