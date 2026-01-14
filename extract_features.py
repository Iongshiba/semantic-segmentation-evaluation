"""
This script extracts features from all images in the Oxford-IIIT Pet dataset
and saves them as CSV files. These CSV files are required for training
traditional machine learning models.

Usage:
    python extract_features.py --dataset_root /path/to/Oxford_IIIT_Pet --image_size 224
"""

import os
import argparse
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from traditional.feature_extractor import FeatureExtractor


def parse_args():
    parser = argparse.ArgumentParser(description='Extract features from Oxford-IIIT Pet dataset')
    parser.add_argument('--dataset_root', type=str, 
                        default='/mnt/c/Users/Admin/Documents/long/document/dataset/Oxford_IIIT_Pet',
                        help='Path to the Oxford-IIIT Pet dataset root directory')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Resize images to this size (default: 224)')
    parser.add_argument('--splits', nargs='+', default=['trainval', 'test'],
                        help='Dataset splits to process (default: trainval test)')
    return parser.parse_args()


def load_and_resize_image(image_path, size):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((size, size), Image.BILINEAR)
    return np.array(img)


def load_and_resize_mask(mask_path, size):
    mask = Image.open(mask_path)
    mask = mask.resize((size, size), Image.NEAREST)
    mask_array = np.array(mask)
    
    # Convert trimap to binary mask (0: background, 1: foreground, 2: boundary -> 1)
    binary_mask = np.where(mask_array == 2, 1, mask_array)
    binary_mask = np.where(binary_mask == 3, 1, binary_mask)
    
    return binary_mask


def extract_features_for_dataset(dataset_root, image_size=224, splits=['trainval', 'test']):
    dataset_root = Path(dataset_root)
    extractor = FeatureExtractor()
    
    # Create output directory for features
    variant_name = f"{image_size}_{image_size}"
    output_dir = dataset_root / 'features' / variant_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting features with image size: {image_size}x{image_size}")
    print(f"Output directory: {output_dir}")
    
    # Process each split
    for split in splits:
        print(f"\n{'='*60}")
        print(f"Processing split: {split}")
        print(f"{'='*60}\n")
        
        # Read the split file
        split_file = dataset_root / 'annotations' / f'{split}.txt'
        
        if not split_file.exists():
            print(f"Warning: Split file not found: {split_file}")
            continue
        
        # Read image names from split file
        image_names = []
        with open(split_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                image_name = parts[0]
                image_names.append(image_name)
        
        print(f"Found {len(image_names)} images in {split} split")
        
        # Process each image
        for image_name in tqdm(image_names, desc=f"Extracting features ({split})"):
            # Check if CSV already exists
            csv_path = output_dir / f"{image_name}.csv"
            if csv_path.exists():
                continue  # Skip if already processed
            
            # Load image and mask
            image_path = dataset_root / 'images' / f"{image_name}.jpg"
            mask_path = dataset_root / 'trimaps' / f"{image_name}.png"
            
            if not image_path.exists():
                print(f"Warning: Image not found: {image_path}")
                continue
            
            if not mask_path.exists():
                print(f"Warning: Mask not found: {mask_path}")
                continue
            
            try:
                # Load and resize
                image = load_and_resize_image(image_path, image_size)
                mask = load_and_resize_mask(mask_path, image_size)
                
                # Extract features
                features_df = extractor.extract_features(image, mask)
                
                # Save to CSV
                features_df.to_csv(csv_path, index=False)
                
            except Exception as e:
                print(f"Error processing {image_name}: {e}")
                continue
        
        print(f"\nCompleted processing {split} split")
    
    print(f"\n{'='*60}")
    print(f"Feature extraction complete!")
    print(f"Features saved to: {output_dir}")
    print(f"{'='*60}\n")


def main():
    args = parse_args()
    
    # Verify dataset root exists
    if not os.path.exists(args.dataset_root):
        print(f"Error: Dataset root not found: {args.dataset_root}")
        print("Please download the Oxford-IIIT Pet dataset and specify the correct path.")
        return
    
    # Run feature extraction
    extract_features_for_dataset(
        dataset_root=args.dataset_root,
        image_size=args.image_size,
        splits=args.splits
    )


if __name__ == '__main__':
    main()
