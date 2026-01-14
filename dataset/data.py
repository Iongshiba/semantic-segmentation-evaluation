import sys
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.oxford_pet import get_oxford_pet_loader
from dataset.oxford_pet_ml import get_oxford_pet_ml_loader
from dataset.rdd2022_seg_coco import get_coco_loader
from dataset.rdd2022_seg_coco_ml import get_rdd2022_ml_loader
from traditional.feature_extractor import FeatureExtractor

# for traditional method
def build_training_data(loader, num_pixels, data_root=None, split_name="train"):
    # Check if cache exists
    cache_file = None
    if data_root is not None:
        cache_file = os.path.join(data_root, f"training_data_cache_{split_name}.csv")
        if os.path.exists(cache_file):
            print(f"Loading cached training data from {cache_file}")
            return pd.read_csv(cache_file)
    
    frames = []
    extractor = FeatureExtractor()
    
    for images, masks, trimaps, bboxes in tqdm(loader, desc="Building dataset"):
        for image, mask in zip(images, masks):
            image_np = image.numpy()
            mask_np = mask.numpy()
            
            image_np = np.transpose(image_np, (1, 2, 0))
            
            feature_df = extractor.extract_features(image_np, mask_np)

            foreground_df = feature_df[feature_df["annotation"] == 1]
            background_df = feature_df[feature_df["annotation"] == 0]

            foreground_num = min(len(foreground_df), num_pixels // 2)
            background_num = min(len(background_df), num_pixels // 2)

            frames.extend([
                foreground_df.sample(foreground_num, random_state=42),
                background_df.sample(background_num, random_state=42),
            ])
    
    result_df = pd.concat(frames, ignore_index=True)
    
    # Save cache if data_root is provided
    if cache_file is not None:
        os.makedirs(data_root, exist_ok=True)
        result_df.to_csv(cache_file, index=False)
        print(f"Cached training data saved to {cache_file}")
    
    return result_df



def get_dataloader(root_dir, split, batch_size, image_size=224, num_workers=4, shuffle=True, normalize=False, seed=42, dataset_type='oxford_pet'):
    if dataset_type == 'oxford_pet':
        return get_oxford_pet_loader(
            root_dir=root_dir,
            split=split,
            batch_size=batch_size,
            image_size=image_size,
            num_workers=num_workers,
            shuffle=shuffle,
            normalize=normalize,
            seed=seed,
        )
    elif dataset_type == 'rdd2022':
        return get_coco_loader(
            root_dir=root_dir,
            split=split,
            batch_size=batch_size,
            image_size=image_size,
            num_workers=num_workers,
            shuffle=shuffle,
            normalize=normalize,
            seed=seed,
        )


def get_ml_dataloader(root_dir, split, batch_size, num_workers=4, shuffle=True, seed=42, dataset_type='oxford_pet', pixels_per_image=50):
    if dataset_type == 'oxford_pet':
        return get_oxford_pet_ml_loader(
            root_dir=root_dir,
            split=split,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            seed=seed,
            pixels_per_image=pixels_per_image,
        )
    elif dataset_type == 'rdd2022':
        return get_rdd2022_ml_loader(
            root_dir=root_dir,
            split=split,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            seed=seed,
        )
