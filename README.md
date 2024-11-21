# MNIST CNN Pipeline

[![ML Pipeline](https://github.com/milindchawre/MNIST-CNN-Pipeline/actions/workflows/ml-pipeline.yml/badge.svg)](https://github.com/milindchawre/MNIST-CNN-Pipeline/actions/workflows/ml-pipeline.yml)

A deep learning pipeline for MNIST digit classification with automated testing and CI/CD integration. The project implements a lightweight CNN architecture optimized for high accuracy while maintaining a small parameter footprint.

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

The model uses a simple and efficient CNN architecture:
- Two convolutional layers
- Batch normalization after each conv layer
- ReLU activations
- MaxPooling layers
- Single hidden fully connected layer
- Minimal dropout (0.05)
- Parameter count under 25,000

Architecture details:
1. First Block:
   - Conv layer (1→8 channels, 3x3 kernel)
   - Batch normalization
   - ReLU activation
   - MaxPool (2x2)

2. Second Block:
   - Conv layer (8→16 channels, 3x3 kernel)
   - Batch normalization
   - ReLU activation
   - MaxPool (2x2)

3. Classification Block:
   - Flatten layer
   - Fully connected (16*5*5 → 32)
   - Dropout (0.05)
   - Output layer (32 → 10)

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
- Minimal data augmentation:
  - Random rotation (±1°)
  - Random affine transforms (translate: ±1%, scale: 99-101%, shear: 0.2°)
- Batch sizes:
  - Training: 8
  - Validation: 64
- Adam optimizer:
  - Learning rate: 0.0005
  - Betas: (0.95, 0.999)
  - No weight decay
- OneCycleLR scheduler:
  - Max learning rate: 0.002
  - Linear annealing
  - 5% warmup
  - div_factor: 5
  - final_div_factor: 50

## Testing

The automated test suite includes five comprehensive tests:

1. Parameter Count Test (`test_model_parameter_count`):
   - Verifies total parameters < 25,000
   - Shows detailed parameter count per layer
   - Ensures model efficiency

2. Input/Output Shape Test (`test_model_input_output_shapes`):
   - Validates input shape (1, 1, 28, 28)
   - Validates output shape (1, 10)
   - Ensures dimensional compatibility

3. Output Properties Test (`test_model_output_properties`):
   - Verifies probability distribution
   - Checks output range [0, 1]
   - Confirms probability sum equals 1

4. Batch Processing Test (`test_model_batch_processing`):
   - Tests multiple batch sizes [1, 16, 32, 64, 128]
   - Verifies batch dimension handling
   - Ensures consistent output shapes

5. Training Test (`test_model_training`):
   - Runs complete training cycle
   - Verifies accuracy > 95%
   - Reports final model statistics

Run all tests with detailed logging:
```bash
python -m pytest test_model.py -v -s
```

## CI/CD Pipeline

GitHub Actions workflow automatically:
- Sets up Python 3.8 environment
- Installs project dependencies
- Runs all five test suites
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

## Development Notes

- CPU-based training for wider compatibility
- Single-epoch optimization
- Minimal regularization for better training accuracy
- Parameter-efficient architecture
- Comprehensive test coverage
- Detailed logging of metrics

## License

MIT License - feel free to use and modify for your projects.
