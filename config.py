UNET_CONFIG = {
    'learning_rate': 1e-4,
    'batch_size': 8,
    'epochs': 50,
    'optimizer': 'adam',
    'weight_decay': 1e-5,
    'in_channels': 3,
    'n_classes': 3,
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
    'probability': True,
}

RANDOM_FOREST_CONFIG = {
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2,
    'n_jobs': -1,
}

DATASET_CONFIG = {
    'dataset_root': '/mnt/c/Users/Admin/Documents/long/document/dataset/Oxford_IIIT_Pet/annotations',
    'split': 'trainval',
    'pixels_per_image': 20000,
}

WANDB_CONFIG = {
    'project_name': 'machine-learning-assignment',
    'entity': None,
}

OUTPUT_DIR = './output'
