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
def test_model_parameter_count(capsys):
    """Test 1: Verify model parameter count is within limits"""
    model = MNISTNet()
    layer_params, total_params = count_parameters(model)
    
    print("\n=== Model Parameter Count Test ===")
    for name, param_count in layer_params:
        print(f"{name:10} : {param_count:,} parameters")
    print("-" * 30)
    print(f"Total Parameters: {total_params:,}")
    print(f"Parameter Limit: 25,000")
    
    assert total_params < 25000, f"Model has {total_params:,} parameters, should be less than 25,000"
    print("Parameter count test: PASSED")

@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_model_input_output_shapes():
    """Test 2: Verify input and output tensor shapes"""
    model = MNISTNet()
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    
    print("\n=== Shape Test ===")
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    
    assert test_input.shape == (1, 1, 28, 28), "Input shape should be (batch_size, 1, 28, 28)"
    assert output.shape == (1, 10), "Output shape should be (batch_size, 10)"
    print("Shape test: PASSED")

@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_model_output_properties():
    """Test 3: Verify model output properties (probabilities)"""
    model = MNISTNet()
    test_input = torch.randn(5, 1, 28, 28)
    output = model(test_input)
    
    print("\n=== Output Properties Test ===")
    print(f"Output sum: {output.exp().sum(dim=1)}")
    print(f"Output range: [{output.exp().min().item():.6f}, {output.exp().max().item():.6f}]")
    
    assert torch.allclose(output.exp().sum(dim=1), torch.ones(5)), "Output probabilities should sum to 1"
    assert (output.exp() >= 0).all(), "Output probabilities should be non-negative"
    assert (output.exp() <= 1).all(), "Output probabilities should be <= 1"
    print("Output properties test: PASSED")

@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_model_batch_processing():
    """Test 4: Verify model handles different batch sizes"""
    model = MNISTNet()
    batch_sizes = [1, 16, 32, 64, 128]
    
    print("\n=== Batch Processing Test ===")
    for batch_size in batch_sizes:
        test_input = torch.randn(batch_size, 1, 28, 28)
        output = model(test_input)
        print(f"Batch size {batch_size}: Input {test_input.shape} -> Output {output.shape}")
        assert output.shape == (batch_size, 10), f"Failed for batch size {batch_size}"
    print("Batch processing test: PASSED")

@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_model_training():
    """Test 5: Verify model training and accuracy"""
    from train import train
    print("\n=== Training Test ===")
    
    try:
        accuracy, model = train()
        layer_params, total_params = count_parameters(model)
        
        print("\n=== Final Model Statistics ===")
        print(f"Total Parameters: {total_params:,}")
        print(f"Achieved Accuracy: {accuracy:.2f}%")
        print(f"Required Accuracy: 95.00%")
        
        assert accuracy > 95.0, f"Model accuracy {accuracy:.2f}% is below 95%"
        print("Training accuracy test: PASSED")
        
    except Exception as e:
        print("\n=== Test Failed ===")
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"]) 