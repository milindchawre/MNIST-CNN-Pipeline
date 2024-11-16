# MNIST CNN Pipeline

A deep learning pipeline for MNIST digit classification with automated testing and CI/CD integration. The project implements a custom CNN architecture optimized for high accuracy while maintaining a small parameter footprint.

## Project Structure 

```
.
├── .github/
│   └── workflows/
│       └── ml-pipeline.yml    # GitHub Actions workflow
├── model/
│   └── mnist_model.py         # CNN model architecture
├── train.py                   # Training script with validation
├── test_model.py             # Automated tests
├── visualize_augmentation.py  # Data augmentation visualization
├── requirements.txt          # Project dependencies
├── .gitignore               # Git ignore rules
└── README.md                # Project documentation
```

## Model Architecture

The model uses a CNN architecture with:
- Three convolutional blocks with feature refinement
- 1x1 convolutions for channel-wise feature mixing
- Residual connection in the third block
- Batch normalization and dropout for regularization
- GELU activation functions
- Parameter count under 25,000

Architecture details:
1. First Block:
   - Double convolution (1→4→8 channels)
   - 1x1 convolution for feature refinement
   - MaxPooling and dropout (0.2)

2. Second Block:
   - Double convolution (8→12→16 channels)
   - 1x1 convolution for feature refinement
   - MaxPooling and dropout (0.2)

3. Third Block:
   - Convolution (16→32 channels)
   - 1x1 convolution with residual connection
   - Dropout (0.3)
   - Final classification layer

## Requirements

- Python 3.8+
- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- pytest >= 7.0.0
- matplotlib >= 3.5.0

## Local Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the model:
```bash
python train.py
```

4. Run tests with detailed logs:
```bash
python -m pytest test_model.py -v -s
```

5. Visualize data augmentation:
```bash
python visualize_augmentation.py
```

## Training Features

- Split training (90%) and validation (10%) sets
- Data augmentation:
  - Random rotation (±5°)
  - Random affine transforms (translate: ±5%, scale: 95-105%, shear: 2°)
  - Random erasing (p=0.1, scale=0.02-0.1)
- Batch sizes:
  - Training: 16
  - Validation: 64
- Adam optimizer:
  - Learning rate: 0.002
  - Weight decay: 1e-4
- OneCycleLR scheduler:
  - Max learning rate: 0.01
  - Cosine annealing
  - 20% warmup

## Testing

The automated test suite verifies:
1. Model Architecture:
   - Input shape compatibility (28x28)
   - Output shape (10 classes)
   - Parameter count (< 25,000)
   - Detailed parameter breakdown per layer

2. Model Performance:
   - Training accuracy
   - Validation accuracy (> 95%)
   - Training stability

## CI/CD Pipeline

GitHub Actions workflow automatically:
- Sets up Python 3.8 environment
- Installs project dependencies
- Runs full test suite
- Uploads trained model as artifact
- Retains artifacts for 14 days

## Model Artifacts

Trained models are saved with detailed naming:
```
mnist_model_YYYYMMDD_HHMMSS_acc{accuracy}.pth
```
- Timestamp for version tracking
- Accuracy included in filename
- Stored in `models/` directory

## Data Augmentation Visualization

The project includes a visualization tool:
- Saves original MNIST image
- Shows applied augmentations
- Helps understand transformation effects
- Outputs stored in 'visualization' directory

## Development Notes

- CPU-based training for wider compatibility
- Single-epoch optimization
- Validation-based model selection
- Parameter-efficient architecture
- Detailed logging of metrics

## GitHub Setup

1. Create a new repository
2. Initialize and push:
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

## License

MIT License - feel free to use and modify for your projects.
