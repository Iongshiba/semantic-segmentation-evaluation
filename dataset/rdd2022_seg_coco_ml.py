import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class RDD2022MLDataset(Dataset):
    def __init__(self, features_dir, split='train'):
        self.features_dir = Path(features_dir)
        self.split = split
        
        self.csv_files = sorted(list(self.features_dir.glob(f"{split}_*.csv")))
        
    def __len__(self):
        return len(self.csv_files)
    
    def __getitem__(self, idx):
        csv_path = self.csv_files[idx]
        features_df = pd.read_csv(csv_path)
        
        X = features_df.drop(columns=['annotation']).values.astype(np.float32)
        y = features_df['annotation'].values.astype(np.long)
        
        return {
            'features': X,
            'annotation': y,
            'file_name': csv_path.stem,
        }


def collate_fn_ml(batch):
    all_features = []
    all_annotations = []
    
    for item in batch:
        all_features.append(item['features'])
        all_annotations.append(item['annotation'])
    
    features = np.vstack(all_features)
    annotations = np.concatenate(all_annotations)
    
    return {
        'features': features,
        'annotation': annotations,
    }


def get_rdd2022_ml_loader(features_dir, split='train', batch_size=32, num_workers=0, shuffle=None, seed=42):
    if shuffle is None:
        shuffle = (split == 'train')
    
    dataset = RDD2022MLDataset(features_dir=features_dir, split=split)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn_ml
    )
    return loader
