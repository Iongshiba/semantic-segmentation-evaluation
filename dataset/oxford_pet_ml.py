import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from tqdm import tqdm


class OxfordPetMLDataset(Dataset):
    def __init__(self, root_dir, split='trainval', variant='224_224', pixels_per_image=50):
        self.root_dir = Path(root_dir)
        self.split = split
        self.variant = variant
        self.pixels_per_image = pixels_per_image
        
        split_file = self.root_dir / 'annotations' / f'{split}.txt'
        self.image_names = []
        
        with open(split_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                image_name = parts[0]
                self.image_names.append(image_name)
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        csv_path = self.root_dir / 'features' / self.variant / f"{image_name}.csv"
        
        features_df = pd.read_csv(csv_path)
        
        if self.pixels_per_image is not None and len(features_df) > self.pixels_per_image:
            features_df = features_df.sample(n=self.pixels_per_image, random_state=42)
        
        X = features_df.drop(columns=['annotation']).values.astype(np.float32)
        y = features_df['annotation'].values.astype(np.long)
        
        return {
            'features': X,
            'annotation': y,
            'image_name': image_name,
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


def get_oxford_pet_ml_loader(root_dir, split='trainval', variant='224_224', batch_size=32, num_workers=0, shuffle=None, seed=42, pixels_per_image=50):
    if shuffle is None:
        shuffle = (split == 'trainval')
    
    dataset = OxfordPetMLDataset(root_dir=root_dir, split=split, variant=variant, pixels_per_image=pixels_per_image)
    
    if split == 'trainval':
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_set, val_set = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn_ml
        )
        val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn_ml
        )
        return train_loader, val_loader
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn_ml
        )
        return loader
