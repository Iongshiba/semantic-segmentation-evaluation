import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# from data import loader
from oxford_pet import get_oxford_pet_loader
from traditional.feature_extractor import FeatureExtractor

# for traditional method
def build_training_data(loader, num_pixels):
    frames = []
    extractor = FeatureExtractor()
    
    for images, masks, indices in tqdm(loader, desc="Building dataset"):
        for i, idx in enumerate(indices):
            sample = loader.dataset.dataset.dataset[idx.item()]
            image = sample['image']
            mask = sample['mask']
            
            h, w = mask.shape
            total_pixels = h * w
            
            if total_pixels <= num_pixels:
                pixel_indices = np.arange(total_pixels)
            else:
                pixel_indices = np.random.choice(total_pixels, num_pixels, replace=False)
            
            feature_df = extractor.extract_features(image, mask)
            sampled_df = feature_df.iloc[pixel_indices]
            frames.append(sampled_df)
    
    return pd.concat(frames, ignore_index=True)


def get_dataloader(root_dir, split, batch_size, image_size=224, num_workers=4, shuffle=True, seed=42):
    return get_oxford_pet_loader(
        root_dir=root_dir,
        split=split,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
        shuffle=shuffle,
        seed=seed,
    )
