import json
import wandb
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import torchvision.transforms.functional as F

from traditional.feature_extractor import FeatureExtractor
from traditional.segmentor import Segmentor
from deep_learning.dice import DiceLoss
from deep_learning.build import UNet
from metrics import compute_metrics
from dataset.data import build_training_data, get_dataloader
from config import (
    UNET_CONFIG, KNN_CONFIG, SVM_CONFIG, RANDOM_FOREST_CONFIG,
    DATASET_CONFIG, OUTPUT_DIR, CHECKPOINT_CONFIG
)



def save_unet_results(model, test_loader, device, metrics_dict, num_samples=10, output_base_dir=OUTPUT_DIR):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_base_dir) / f"unet_test_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    dataset = test_loader.dataset
    num_samples = min(num_samples, len(dataset))
    
    all_predictions = []
    sample_indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)
    
    with torch.no_grad():
        for idx in sample_indices:
            sample = dataset[idx]
            image_tensor = sample['image'].unsqueeze(0).float().to(device)
            image_tensor = F.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            
            output = model(image_tensor)
            predicted_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            
            all_predictions.append(predicted_mask)
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].axis("off")
            axes[0].set_title("Original Image")
            axes[0].imshow(sample['image'].permute(1, 2, 0).numpy())
            
            axes[1].axis("off")
            axes[1].set_title("Ground Truth Mask")
            axes[1].imshow(sample['mask'], cmap='gray')
            
            axes[2].axis("off")
            axes[2].set_title("Predicted Segmentation")
            axes[2].imshow(predicted_mask, cmap='gray')
            
            plt.tight_layout()
            plt.savefig(output_dir / f"sample_{idx}.png", dpi=150, bbox_inches='tight')
            plt.close()
    
    np.save(output_dir / "predictions.npy", np.array(all_predictions))
    
    results_json = {
        'model_type': 'unet',
        'timestamp': timestamp,
        'num_samples': num_samples,
        'test_metrics': metrics_dict,
    }
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results_json, indent=2, fp=f)
    
    return output_dir

def save_traditional_results(model, test_loader, metrics_dict, num_samples=10, output_base_dir=OUTPUT_DIR):    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    time.sleep(0.1)
    millis = int(time.time() * 1000) % 1000
    output_dir = Path(output_base_dir) / f"{model.classifier_type}_{timestamp}_{millis}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extractor = FeatureExtractor()
    dataset = test_loader.dataset
    num_samples = min(num_samples, len(dataset))
    sample_indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)
    
    all_predictions = []
    for idx in sample_indices:
        idx = int(idx)
        sample = dataset[idx]
        image = sample['image']
        mask = sample['mask']
        
        image_np = image.numpy()
        mask_np = mask.numpy()
        image_np = np.transpose(image_np, (1, 2, 0))
        
        feature_df = extractor.extract_features(image_np, mask_np)
        X = feature_df.drop(columns=['annotation']).values
        X_scaled = model.scaler.transform(X)
        predicted = model.predict(X_scaled).reshape(mask.shape)
        all_predictions.append(predicted)
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].axis("off")
        axes[0].set_title("Ground Truth Mask")
        axes[0].imshow(mask, cmap='gray')
        
        axes[1].axis("off")
        axes[1].set_title("Predicted Segmentation")
        axes[1].imshow(predicted, cmap='gray')
        
        plt.tight_layout()
        plt.savefig(output_dir / f"sample_{idx}.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    np.save(output_dir / "predictions.npy", np.array(all_predictions))
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump({
            'model_type': model.classifier_type,
            'sample': DATASET_CONFIG['pixels_per_image'],
            'timestamp': timestamp,
            'test_metrics': metrics_dict,
        }, f, indent=2)
    
    return output_dir



def evaluate_unet_model(unet, dl_loader, device):
    metrics = evaluate_unet(unet, dl_loader, criterion=None, device=device)

    wandb.log({
        f'unet/test_iou': metrics['iou'],
        f'unet/test_dice': metrics['dice'],
        f'unet/test_precision': metrics['precision'],
        f'unet/test_recall': metrics['recall'],
    })

    return metrics

def evaluate_traditional_models(models, test_loader):
    results = {}
    
    for model_name, model in models.items():
        metrics = evaluate_traditional_model(model, test_loader)
        results[model_name] = metrics
        
        wandb.log({
            f'{model_name}/test_iou': metrics['iou'],
            f'{model_name}/test_dice': metrics['dice'],
            f'{model_name}/test_precision': metrics['precision'],
            f'{model_name}/test_recall': metrics['recall'],
        })

    return results



def evaluate_unet(model, loader, criterion=None, device='cuda'):
    model.eval()
    all_predictions = []
    all_ground_truth = []
    total_loss = 0.0
    
    with torch.no_grad():
        for images, masks, trimaps, bboxes in tqdm(loader, desc="Evaluating UNet"):
            images = images.float().to(device)
            images = F.normalize(images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            masks = masks.to(device)
            
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            
            if criterion:
                ce, dice = criterion
                loss = ce(outputs, masks) + dice(outputs, masks)
                total_loss += loss.item()
            
            for pred, gt in zip(predictions, masks.cpu().numpy()):
                all_predictions.append(pred)
                all_ground_truth.append(gt)

    all_predictions = np.array(all_predictions)
    all_ground_truth = np.array(all_ground_truth)
    
    metrics = compute_metrics(all_predictions, all_ground_truth)
    
    if criterion:
        avg_loss = total_loss / len(loader)
        metrics['loss'] = avg_loss

    return metrics

def train_unet_model(dl_train_loader, dl_val_loader, device, output_dir=None):
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(OUTPUT_DIR) / f"unet_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(output_dir)
    
    unet = UNet(
        n_channels=UNET_CONFIG['in_channels'],
        n_classes=UNET_CONFIG['n_classes'],
        dropout=UNET_CONFIG.get('dropout', 0.3)
    ).to(device)
    
    class_weights = torch.tensor(UNET_CONFIG.get('class_weights', [1.0, 3.0])).to(device)
    ce = nn.CrossEntropyLoss(weight=class_weights)
    dice = DiceLoss()
    dice_weight = UNET_CONFIG.get('dice_weight', 2.0)
    
    optimizer = optim.Adam(
        unet.parameters(),
        lr=UNET_CONFIG['learning_rate'],
        weight_decay=UNET_CONFIG['weight_decay']
    )
    
    checkpoint_freq = UNET_CONFIG.get('checkpoint_freq', CHECKPOINT_CONFIG.get('save_freq', 5))
    
    total_params = sum(p.numel() for p in unet.parameters())
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    
    wandb.config.update({
        'unet/total_parameters': total_params,
        'unet/trainable_parameters': trainable_params,
        'unet/device': str(device),
        'unet/output_dir': str(output_dir),
        'unet/train_batches': len(dl_train_loader),
        'unet/val_batches': len(dl_val_loader),
        'unet/class_weights': class_weights.cpu().tolist(),
        'unet/dice_weight': dice_weight,
    })
    
    for epoch in range(UNET_CONFIG['epochs']):
        unet.train()
        total_loss = 0
        pbar = tqdm(dl_train_loader, desc=f'Epoch {epoch}')

        for batch_idx, (images, masks, trimaps, bboxes) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = unet(images)
            loss = ce(outputs, masks) + dice_weight * dice(outputs, masks)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(dl_train_loader)
        
        unet_metrics = evaluate_unet(unet, dl_val_loader, (ce, dice), device)
        wandb.log({
            'unet/train_loss': avg_loss,
            'unet/val_loss': unet_metrics['loss'],
            'unet/val_iou': unet_metrics['iou'],
            'unet/val_dice': unet_metrics['dice'],
            'unet/val_precision': unet_metrics['precision'],
            'unet/val_recall': unet_metrics['recall'],
            'unet/epoch': epoch
        })
        
        if (epoch + 1) % checkpoint_freq == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'metrics': unet_metrics,
            }, checkpoint_path)

    final_path = output_dir / "final_model.pth"
    torch.save(unet.state_dict(), final_path)
    
    return unet



def evaluate_traditional_model(model, features_loader):
    all_predictions = []
    all_ground_truth = []
    
    for batch in tqdm(features_loader, desc="Evaluating"):
        X = batch['features']
        y = batch['annotation']
        
        X_scaled = model.scaler.transform(X)
        predictions = model.predict(X_scaled)
        
        all_predictions.append(predictions)
        all_ground_truth.append(y)
    
    all_predictions = np.concatenate(all_predictions)
    all_ground_truth = np.concatenate(all_ground_truth)
    
    return compute_metrics(all_predictions, all_ground_truth)

def train_traditional_models(train_loader, val_loader, output_dir=None):
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(OUTPUT_DIR) / f"traditional_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(output_dir)
    
    train_features = []
    train_annotations = []
    
    for batch in tqdm(train_loader, desc="Loading ML features"):
        train_features.append(batch['features'])
        train_annotations.append(batch['annotation'])
    
    train_X = np.vstack(train_features)
    train_y = np.concatenate(train_annotations)
    train_df = pd.DataFrame(train_X)
    train_df['annotation'] = train_y

    print(f"{'=' * 30} Start training traditional ML {'=' * 30}")
    
    wandb.config.update({
        'traditional/train_samples': len(train_df),
        'traditional/num_features': train_X.shape[1],
        'traditional/output_dir': str(output_dir),
    })
    
    models = {}
    
    svm = Segmentor("svm", config=SVM_CONFIG)
    svm.fit_predict(train_df)
    svm_metrics = evaluate_traditional_model(svm, val_loader)
    wandb.log({
        'svm/iou': svm_metrics['iou'],
        'svm/dice': svm_metrics['dice'],
        'svm/precision': svm_metrics['precision'],
        'svm/recall': svm_metrics['recall'],
    })
    models['svm'] = svm
    with open(output_dir / 'svm_model.pkl', 'wb') as f:
        pickle.dump(svm, f)
    
    knn = Segmentor("knn", config=KNN_CONFIG)
    knn.fit_predict(train_df)
    knn_metrics = evaluate_traditional_model(knn, val_loader)
    wandb.log({
        'knn/iou': knn_metrics['iou'],
        'knn/dice': knn_metrics['dice'],
        'knn/precision': knn_metrics['precision'],
        'knn/recall': knn_metrics['recall'],
    })
    models['knn'] = knn
    with open(output_dir / 'knn_model.pkl', 'wb') as f:
        pickle.dump(knn, f)
    
    rf = Segmentor("random_forest", config=RANDOM_FOREST_CONFIG)
    rf.fit_predict(train_df)
    rf_metrics = evaluate_traditional_model(rf, val_loader)
    wandb.log({
        'random_forest/iou': rf_metrics['iou'],
        'random_forest/dice': rf_metrics['dice'],
        'random_forest/precision': rf_metrics['precision'],
        'random_forest/recall': rf_metrics['recall'],
    })
    models['random_forest'] = rf
    with open(output_dir / 'random_forest_model.pkl', 'wb') as f:
        pickle.dump(rf, f)

    return models
