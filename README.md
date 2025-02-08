# Rickets Detection Using Deep Learning

## Overview
This project implements a deep learning model for detecting rickets from medical images using transfer learning with EfficientNetB0. The model is designed to classify images into two categories: normal and rickets cases.

## Features
- Transfer learning using EfficientNetB0 architecture
- Image preprocessing with CLAHE enhancement
- Data augmentation for improved model robustness
- Handling of class imbalance
- Support for various image formats (RGB, RGBA, Grayscale)
- Real-time prediction visualization
- Model checkpointing and early stopping
- Google Drive integration for data storage

## Prerequisites
- Python 3.7+
- Google Colab (recommended for training)
- Google Drive for dataset storage

## Project Structure
```
rickets_dataset/
├── normal/
│   └── [normal case images].png
├── rickets/
│   └── [rickets case images].png
├── best_model.keras
└── rickets_model_final.keras
```

## Installation
1. Clone this repository
2. Mount Google Drive in Colab
3. Install required dependencies
4. Upload your dataset to the specified Google Drive directory

## Usage

### Dataset Preparation
1. Organize your dataset into two folders:
   - `normal/` - containing normal case images
   - `rickets/` - containing rickets case images
2. Place all images in PNG format

### Training
```python
# Initialize model
model = RicketsPredictionModel()

# Check and prepare dataset
model.check_dataset(data_dir)
model.create_model()

# Train model
images, labels = model.prepare_data(data_dir)
X_train, X_val, y_train, y_val = train_test_split(
    images, labels,
    test_size=0.15,
    random_state=42,
    stratify=labels
)
model.train_model(X_train, y_train, X_val, y_val)
```

### Making Predictions
```python
# Predict on a single image
prediction = predict_from_drive('image_name.png')
```

## Model Architecture
- Base model: EfficientNetB0 (pretrained on ImageNet)
- Additional layers:
  - Global Average Pooling
  - Batch Normalization
  - Dense layers with dropout for regularization
  - Binary classification output

## Training Features
- Data augmentation (rotation, flip, zoom, brightness adjustment)
- Class weight balancing
- Learning rate scheduling
- Early stopping
- Model checkpointing

## Performance Monitoring
- Training/validation accuracy
- AUC-ROC curve
- Real-time visualization of predictions

## Acknowledgments
- TensorFlow team for the EfficientNet implementation
- Google Colab for providing GPU resources
