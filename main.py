import os
import torch
import wandb

from config import (
    UNET_CONFIG, KNN_CONFIG, SVM_CONFIG, RANDOM_FOREST_CONFIG,
    DATASET_CONFIG, WANDB_CONFIG
)
from dataset.data import get_dataloader, get_ml_dataloader
from engine import (
    train_unet_model, train_traditional_models,
    evaluate_unet_model, evaluate_traditional_models,
    save_unet_results, save_traditional_results,
)


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
    
    dl_train_loader, dl_val_loader = get_dataloader(
        root_dir=DATASET_CONFIG['dataset_root'],
        split=DATASET_CONFIG['train'],
        batch_size=UNET_CONFIG['batch_size'],
        image_size=224,
        num_workers=4,
        shuffle=True,
        normalize=True,
        seed=42,
    )
    
    trad_train_loader, trad_val_loader = get_ml_dataloader(
        root_dir=DATASET_CONFIG['dataset_root'],
        split=DATASET_CONFIG['train'],
        batch_size=UNET_CONFIG['batch_size'],
        pixels_per_image=DATASET_CONFIG['pixels_per_image'],
        num_workers=4,
        shuffle=True,
        seed=42,
    )

    test_loader = get_dataloader(
        root_dir=DATASET_CONFIG['dataset_root'],
        split=DATASET_CONFIG['test'],
        batch_size=UNET_CONFIG['batch_size'],
        image_size=224,
        num_workers=4,
        shuffle=True,
        seed=42,
    )
    
    test_ml_loader = get_ml_dataloader(
        root_dir=DATASET_CONFIG['dataset_root'],
        split=DATASET_CONFIG['test'],
        batch_size=UNET_CONFIG['batch_size'],
        pixels_per_image=DATASET_CONFIG['pixels_per_image'],
        num_workers=4,
        shuffle=True,
        seed=42,
    )
    
    unet = train_unet_model(dl_train_loader, dl_val_loader, device)
    test_unet_metrics = evaluate_unet_model(unet, test_loader, device)
    save_unet_results(unet, test_loader, device, test_unet_metrics)

    # traditional = train_traditional_models(trad_train_loader, trad_val_loader)
    # test_traditional_metrics = evaluate_traditional_models(traditional, test_ml_loader)
    # for model_name, model in traditional.items():
    #     save_traditional_results(model, test_loader, test_traditional_metrics[model_name])
    
    wandb.finish()

if __name__ == "__main__":
    main()

