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
        transforms.RandomRotation(5),
        transforms.RandomAffine(
            degrees=5,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
            shear=2
        ),
        transforms.RandomErasing(p=1.0, scale=(0.02, 0.1))  # Set p=1.0 to ensure erasing happens
    ])
    
    # Apply augmentation
    augmented_image = augmentation(original_image)
    
    # Save augmented image
    plt.figure(figsize=(4, 4))
    plt.imshow(augmented_image.squeeze(), cmap='gray')
    plt.axis('off')
    plt.savefig('visualization/augmented-image.png', bbox_inches='tight', pad_inches=0)
    plt.close()

if __name__ == "__main__":
    save_augmented_samples()
    print("Images saved in 'visualization' directory:")
    print("1. original-image.png")
    print("2. augmented-image.png") 