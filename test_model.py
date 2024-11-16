import torch
from model.mnist_model import MNISTNet
from torchvision import datasets, transforms
import pytest

def count_parameters(model):
    """Count parameters for each layer and total"""
    params_per_layer = [(name, sum(p.numel() for p in layer.parameters() if p.requires_grad))
                       for name, layer in model.named_children()]
    total_params = sum(count for _, count in params_per_layer)
    return params_per_layer, total_params

@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_model_architecture(capsys):
    model = MNISTNet()
    layer_params, total_params = count_parameters(model)
    
    # Always print parameter counts
    print("\n=== Model Architecture Details ===")
    for name, param_count in layer_params:
        print(f"{name:10} : {param_count:,} parameters")
    print("-" * 30)
    print(f"Total Parameters: {total_params:,}")
    print(f"Parameter Limit: 25,000")
    print(f"Status: {'PASSED' if total_params < 25000 else 'FAILED'}")
    
    # Test input shape
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    print(f"\nInput shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test output shape and parameters
    assert output.shape == (1, 10), "Output shape should be (batch_size, 10)"
    assert total_params < 25000, f"Model has {total_params:,} parameters, should be less than 25,000"

@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_model_training(capsys):
    from train import train
    print("\n=== Starting Model Training Test ===")
    
    try:
        accuracy, model = train()
        layer_params, total_params = count_parameters(model)
        
        # Always print final statistics
        print("\n=== Final Model Statistics ===")
        print(f"Total Parameters: {total_params:,}")
        print(f"Achieved Accuracy: {accuracy:.2f}%")
        print(f"Required Accuracy: 95.00%")
        print(f"Status: {'PASSED' if accuracy > 95.0 else 'FAILED'}")
        
        assert accuracy > 95.0, f"Model accuracy {accuracy:.2f}% is below 95%"
        
    except Exception as e:
        # Even if test fails, print the statistics
        print("\n=== Test Failed ===")
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"]) 