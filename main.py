import torch
import wandb

from config import (
    UNET_CONFIG, KNN_CONFIG, SVM_CONFIG, RANDOM_FOREST_CONFIG,
    DATASET_CONFIG, WANDB_CONFIG
)
from data import get_dataloader
from train import (
    train_unet_model, train_traditional_models,
    evaluate_unet_model, evaluate_traditional_models
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
    
    unet = train_unet_model(dl_loader, device)
    evaluate_unet_model(unet, dl_loader, device)
    
    traditional_models = train_traditional_models(trad_loader)
    evaluate_traditional_models(traditional_models)
    
    wandb.finish()

if __name__ == "__main__":
    main()

