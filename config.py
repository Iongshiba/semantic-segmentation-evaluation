UNET_CONFIG = {
    'learning_rate': 1e-5,
    'batch_size': 16,
    'epochs': 500,
    'optimizer': 'auto',
    'weight_decay': 0,
    'in_channels': 3,
    'n_classes': 2,
    'checkpoint_freq': 5,
    'dropout': 0.0,
    'dice_weight': 0.0,
    'class_weights': [1.0, 1.0],
}

KNN_CONFIG = {
    'n_neighbors': 5,
    'weights': 'uniform',
    'algorithm': 'auto',
}

SVM_CONFIG = {
    'kernel': 'rbf',
    'C': 1.0,
    'gamma': 'scale',
    'probability': False,
}

RANDOM_FOREST_CONFIG = {
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2,
    'n_jobs': -1,
}

DATASET_CONFIG = {
    'dataset_root': '/mnt/c/Users/Admin/Documents/long/document/dataset/Oxford_IIIT_Pet',
    'train': 'trainval',
    'val': 'trainval',
    'test': 'test',
    'pixels_per_image': 25,
}

WANDB_CONFIG = {
    'project_name': 'machine-learning-assignment',
    'entity': None,
}

OUTPUT_DIR = './output'

CHECKPOINT_CONFIG = {
    'save_freq': 5,
}
