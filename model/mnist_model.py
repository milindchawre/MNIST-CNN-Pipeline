import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        # First conv layer without padding
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)  # 28x28 -> 26x26
        self.bn1 = nn.BatchNorm2d(8)
        
        # Second conv layer without padding
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)  # 13x13 -> 11x11
        self.bn2 = nn.BatchNorm2d(16)
        
        # Fully connected layers back to original size
        self.fc1 = nn.Linear(16 * 5 * 5, 32)  # Back to 32 units
        self.fc2 = nn.Linear(32, 10)
        
        self.dropout = nn.Dropout(0.05)  # Kept reduced dropout rate

    def forward(self, x):
        # First block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Second block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Fully connected layers
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)