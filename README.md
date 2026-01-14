# Machine Learning Assignment: Image Segmentation

This project implements both deep learning (U-Net) and traditional machine learning approaches (KNN, SVM, Random Forest) for image segmentation on the Oxford-IIIT Pet Dataset.

## Setup Instructions

### 1. Install Dependencies

Install all required Python packages using the provided requirements file:

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

Download the Oxford-IIIT Pet Dataset from Kaggle:

**Dataset Link:** https://www.robots.ox.ac.uk/~vgg/data/pets/

1. Download the dataset from the link above
2. Unzip the downloaded file into your working directory
3. The dataset should be extracted to a directory named `Oxford_IIIT_Pet`

Your directory structure should look like:
```
Oxford_IIIT_Pet/
├── annotations/
│   ├── trainval.txt
│   ├── test.txt
│   └── ...
├── images/
│   └── *.jpg
└── trimaps/
    └── *.png
```

### 3. Extract Features from Dataset

For the traditional ML models, you need to extract features from all images in the dataset. Run the feature extraction script:

```bash
python extract_features.py --dataset_root /path/to/Oxford_IIIT_Pet --image_size 224
```

This will:
- Process all images in the trainval and test splits
- Extract various features (Gabor, edge detectors, filters, etc.) for each pixel
- Save feature CSV files to `Oxford_IIIT_Pet/features/224_224/`
- This may take some time depending on your hardware

Optional arguments:
- `--dataset_root`: Path to your Oxford-IIIT Pet dataset (default: configured path)
- `--image_size`: Image resize dimension (default: 224)
- `--splits`: Dataset splits to process (default: trainval test)

Your final structure will include:
```
Oxford_IIIT_Pet/
├── annotations/
├── images/
├── trimaps/
└── features/
    └── 224_224/
        ├── Abyssinian_1.csv
        ├── Abyssinian_2.csv
        └── ... (one CSV per image)
```

**Note**: If you already have pre-extracted feature CSV files, you can skip this step and place them directly in the `features/224_224/` directory.

### 4. Configure Dataset Path

Update the dataset path in `config.py`:

```python
DATASET_CONFIG = {
    'dataset_root': '/path/to/your/Oxford_IIIT_Pet',  # Update this path
    'train': 'trainval',
    'val': 'trainval',
    'test': 'test',
    'pixels_per_image': 25,
}
```

### 5. Run Training

Execute the main training script:

```bash
python main.py
```

This will:
- Train the U-Net model for image segmentation
- Train traditional ML models (KNN, SVM, Random Forest)
- Log results to Weights & Biases (wandb)
- Save model checkpoints and results to the `output/` directory

## Project Structure

```
.
├── config.py                    # Configuration file for models and dataset
├── main.py                      # Main training script
├── extract_features.py          # Feature extraction script for traditional ML
├── engine.py                    # Training and evaluation logic
├── metrics.py                   # Evaluation metrics
├── dataset/                     # Dataset loaders
│   ├── oxford_pet.py           # Deep learning dataset loader
│   └── oxford_pet_ml.py        # Traditional ML dataset loader
├── deep_learning/              # Deep learning models
│   ├── unet.py                 # U-Net architecture
│   ├── dice.py                 # Dice loss
│   └── build.py                # Model builder
├── traditional/                # Traditional ML components
│   ├── feature_extractor.py   # Feature extraction methods
│   ├── feature_reduction.py   # Dimensionality reduction
│   └── segmentor.py           # ML classifiers
└── output/                     # Training outputs and checkpoints
```

## Configuration

Edit `config.py` to customize:
- U-Net hyperparameters (learning rate, batch size, epochs)
- Traditional ML model parameters
- Dataset settings
- Weights & Biases configuration

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for U-Net training)
- See `requirements.txt` for complete package list

## Outputs

Model checkpoints of U-Net from the report can be found [https://drive.google.com/drive/folders/1raR_YBsJiNPFg182DXPcjswMOxP-pTvk?usp=sharing](https://drive.google.com/drive/folders/1raR_YBsJiNPFg182DXPcjswMOxP-pTvk?usp=sharing)

Training outputs are saved to the `output/` directory:
- Model checkpoints
- Training metrics and logs
- Prediction visualizations
- Evaluation results

## Monitoring

Training progress is logged to Weights & Biases. Configure your wandb settings in `config.py`:

```python
WANDB_CONFIG = {
    'project_name': 'machine-learning-assignment',
    'entity': None,  # Set your wandb username/team
}
```
