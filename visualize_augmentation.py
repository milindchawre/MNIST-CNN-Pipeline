import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

def save_augmented_samples():
    # Create output directory if it doesn't exist
    os.makedirs('visualization', exist_ok=True)
    
    # Load a single MNIST image
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
    original_image = dataset[0][0]  # Get the first image
    
    # Save original image
    plt.figure(figsize=(4, 4))
    plt.imshow(original_image.squeeze(), cmap='gray')
    plt.axis('off')
    plt.savefig('visualization/original-image.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # Define the same augmentation pipeline used in training
    augmentation = transforms.Compose([
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomRotation(1),  # Minimal rotation
        transforms.RandomAffine(
            degrees=1,
            translate=(0.01, 0.01),
            scale=(0.99, 1.01),
            shear=0.2
        )
    ])
    
    # Apply augmentation
    augmented_image = augmentation(original_image)
    
    # Save augmented image
    plt.figure(figsize=(4, 4))
    plt.imshow(augmented_image.squeeze(), cmap='gray')
    plt.axis('off')
    plt.savefig('visualization/augmented-image.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    
    print("Images saved in 'visualization' directory:")
    print("1. original-image.png - Original MNIST digit")
    print("2. augmented-image.png - After applying training augmentations:")
    print("   - Random rotation (±1°)")
    print("   - Random affine transforms:")
    print("     * Translation: ±1%")
    print("     * Scale: 99-101%")
    print("     * Shear: 0.2°")
    print("   - Normalization")

if __name__ == "__main__":
    save_augmented_samples() 