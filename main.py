import cv2
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import wandb

sys.path.append('/mnt/c/Users/Admin/Documents/long/document/dataset/Oxford_IIIT_Pet')

from oxford_pet_loader import get_oxford_pet_loader
from traditional.feature_extractor import FeatureExtractor
from traditional.segmentor import Segmentor
from deep_learning.build import UNet
from config import (
    UNET_CONFIG, KNN_CONFIG, SVM_CONFIG, RANDOM_FOREST_CONFIG,
    DATASET_CONFIG, WANDB_CONFIG, OUTPUT_DIR
)
from metrics import compute_metrics
from data import build_training_data, get_dataloader


def train_unet(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for images, masks, trimaps, bboxes in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(train_loader)


def evaluate_unet(model, loader, device):
    model.eval()
    all_predictions = []
    all_ground_truth = []
    
    with torch.no_grad():
        for images, masks, trimaps, bboxes in tqdm(loader, desc="Evaluating UNet"):
            images = images.to(device)
            masks = masks.cpu().numpy()
            
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            
            for pred, gt in zip(predictions, masks):
                all_predictions.append(pred)
                all_ground_truth.append(gt)
    
    all_predictions = np.array(all_predictions)
    all_ground_truth = np.array(all_ground_truth)
    
    return compute_metrics(all_predictions, all_ground_truth)


def save_unet_results(model, dataset, device, metrics_dict, output_base_dir=OUTPUT_DIR):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_base_dir) / f"unet_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    sample_idx = 0
    sample = dataset.dataset.dataset[sample_idx]
    
    image_tensor = torch.from_numpy(sample['image']).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    
    with torch.no_grad():
        output = model(image_tensor)
        predicted_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].axis("off")
    axes[0].set_title("Original Image")
    axes[0].imshow(sample['image'])
    
    axes[1].axis("off")
    axes[1].set_title("Ground Truth Mask")
    axes[1].imshow(sample['mask'], cmap='gray')
    
    axes[2].axis("off")
    axes[2].set_title("Predicted Segmentation")
    axes[2].imshow(predicted_mask, cmap='gray')
    
    plt.tight_layout()
    plt.savefig(output_dir / "predicted_segmentation.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    torch.save(model.state_dict(), output_dir / "model.pth")
    np.save(output_dir / "predicted_mask.npy", predicted_mask)
    
    results_json = {
        'model_type': 'unet',
        'timestamp': timestamp,
        'image_name': sample['image_name'],
        'image_shape': list(sample['mask'].shape),
        'config': UNET_CONFIG,
        'metrics': metrics_dict,
    }
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results_json, indent=2, fp=f)
    
    return output_dir


def save_traditional_results(segmentor, results, feature_df, sample, metrics_dict, output_base_dir=OUTPUT_DIR):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_base_dir) / f"{segmentor.classifier_type}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_shape = sample['mask'].shape
    
    extractor = FeatureExtractor()
    full_features = extractor.extract_features(sample['image'], sample['mask'])
    X_full = full_features.drop(columns=['annotation']).values
    X_scaled = segmentor.scaler.transform(X_full)
    predictions = segmentor.predict(X_scaled)
    predicted_mask = predictions.reshape(image_shape)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].axis("off")
    axes[0].set_title("Original Image")
    axes[0].imshow(sample['image'])
    
    axes[1].axis("off")
    axes[1].set_title("Ground Truth Mask")
    axes[1].imshow(sample['mask'], cmap='gray')
    
    axes[2].axis("off")
    axes[2].set_title("Predicted Segmentation")
    axes[2].imshow(predicted_mask, cmap='gray')
    
    plt.tight_layout()
    plt.savefig(output_dir / "predicted_segmentation.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    np.save(output_dir / "predicted_mask.npy", predicted_mask)
    
    with open(output_dir / "model.pkl", 'wb') as f:
        pickle.dump(segmentor.model, f)
    
    with open(output_dir / "scaler.pkl", 'wb') as f:
        pickle.dump(segmentor.scaler, f)
    
    results_json = {
        'classifier_type': segmentor.classifier_type,
        'timestamp': timestamp,
        'image_name': sample['image_name'],
        'image_shape': list(image_shape),
        'num_features': len(feature_df.columns) - 1,
        'num_samples': len(feature_df),
        'accuracy': float(results['metrics']['accuracy']),
        'classification_report': results['metrics']['classification_report'],
        'confusion_matrix': results['metrics']['confusion_matrix'].tolist(),
        'segmentation_metrics': metrics_dict,
    }
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results_json, indent=2, fp=f)
    
    return output_dir


def main():
    wandb.init(
        project=WANDB_CONFIG['project_name'],
        entity=WANDB_CONFIG['entity'],
        config={
            'unet': UNET_CONFIG,
            'knn': KNN_CONFIG,
            'svm': SVM_CONFIG,
            'random_forest': RANDOM_FOREST_CONFIG,
            'dataset': DATASET_CONFIG,
        }
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dl_loader = get_dataloader(
        root_dir=DATASET_CONFIG['dataset_root'],
        split=DATASET_CONFIG['split'],
        batch_size=UNET_CONFIG['batch_size'],
        image_size=224,
        num_workers=4,
        shuffle=True,
        seed=42,
    )
    
    trad_loader = get_dataloader(
        root_dir=DATASET_CONFIG['dataset_root'],
        split=DATASET_CONFIG['split'],
        batch_size=UNET_CONFIG['batch_size'],
        num_workers=4,
        shuffle=True,
        seed=42,
    )
    
    unet = UNet(
        n_channels=UNET_CONFIG['in_channels'],
        n_classes=UNET_CONFIG['n_classes']
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        unet.parameters(),
        lr=UNET_CONFIG['learning_rate'],
        weight_decay=UNET_CONFIG['weight_decay']
    )
    
    for epoch in range(UNET_CONFIG['epochs']):
        loss = train_unet(unet, dl_loader, criterion, optimizer, device, epoch)
    
    unet_metrics = evaluate_unet(unet, dl_loader, device)
    
    wandb.log({
        'unet/iou': unet_metrics['iou'],
        'unet/dice': unet_metrics['dice'],
        'unet/precision': unet_metrics['precision'],
        'unet/recall': unet_metrics['recall']
    })
    
    save_unet_results(unet, dl_loader, device, unet_metrics)
    
    feature_df = build_training_data(trad_loader, DATASET_CONFIG['pixels_per_image'])
    
    sample = trad_loader.dataset.dataset.dataset[0]
    
    svm = Segmentor("svm", config=SVM_CONFIG)
    svm_results = svm.fit_predict(feature_df)
    
    extractor = FeatureExtractor()
    full_features = extractor.extract_features(sample['image'], sample['mask'])
    X_full = full_features.drop(columns=['annotation']).values
    X_scaled = svm.scaler.transform(X_full)
    svm_predictions = svm.predict(X_scaled).reshape(sample['mask'].shape)
    svm_metrics = compute_metrics(svm_predictions, sample['mask'])
    
    wandb.log({
        'svm/iou': svm_metrics['iou'],
        'svm/dice': svm_metrics['dice'],
        'svm/precision': svm_metrics['precision'],
        'svm/recall': svm_metrics['recall']
    })
    
    save_traditional_results(svm, svm_results, feature_df, sample, svm_metrics)
    
    knn = Segmentor("knn", config=KNN_CONFIG)
    knn_results = knn.fit_predict(feature_df)
    
    X_scaled_knn = knn.scaler.transform(X_full)
    knn_predictions = knn.predict(X_scaled_knn).reshape(sample['mask'].shape)
    knn_metrics = compute_metrics(knn_predictions, sample['mask'])
    
    wandb.log({
        'knn/iou': knn_metrics['iou'],
        'knn/dice': knn_metrics['dice'],
        'knn/precision': knn_metrics['precision'],
        'knn/recall': knn_metrics['recall']
    })
    
    save_traditional_results(knn, knn_results, feature_df, sample, knn_metrics)
    
    rf = Segmentor("random_forest", config=RANDOM_FOREST_CONFIG)
    rf_results = rf.fit_predict(feature_df)
    
    X_scaled_rf = rf.scaler.transform(X_full)
    rf_predictions = rf.predict(X_scaled_rf).reshape(sample['mask'].shape)
    rf_metrics = compute_metrics(rf_predictions, sample['mask'])
    
    wandb.log({
        'random_forest/iou': rf_metrics['iou'],
        'random_forest/dice': rf_metrics['dice'],
        'random_forest/precision': rf_metrics['precision'],
        'random_forest/recall': rf_metrics['recall']
    })
    
    save_traditional_results(rf, rf_results, feature_df, sample, rf_metrics)
    
    wandb.finish()

if __name__ == "__main__":
    main()
