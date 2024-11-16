import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        # First conv block with two conv layers
        self.conv1a = nn.Conv2d(1, 4, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.conv1b = nn.Conv2d(4, 8, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.conv1c = nn.Conv2d(8, 8, kernel_size=1)  # 1x1 conv for feature refinement
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(2)  # 28x28 -> 14x14
        
        # Second conv block with two conv layers
        self.conv2a = nn.Conv2d(8, 12, kernel_size=3, padding=1)  # 14x14 -> 14x14
        self.conv2b = nn.Conv2d(12, 16, kernel_size=3, padding=1)  # 14x14 -> 14x14
        self.conv2c = nn.Conv2d(16, 16, kernel_size=1)  # 1x1 conv for feature refinement
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2)  # 14x14 -> 7x7
        
        # Third conv block with residual connection
        self.conv3a = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 7x7 -> 7x7
        self.conv3b = nn.Conv2d(32, 32, kernel_size=1)  # 1x1 conv for feature refinement
        self.bn3 = nn.BatchNorm2d(32)
        
        # Fully connected layer
        self.fc = nn.Linear(32 * 7 * 7, 10)
        
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.3)

    def forward(self, x):
        # First double conv block
        x = F.gelu(self.conv1a(x))
        x = F.gelu(self.conv1b(x))
        x = self.conv1c(x)  # 1x1 conv
        x = self.bn1(x)
        x = F.gelu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second double conv block
        x = F.gelu(self.conv2a(x))
        x = F.gelu(self.conv2b(x))
        x = self.conv2c(x)  # 1x1 conv
        x = self.bn2(x)
        x = F.gelu(x)
        x = self.pool2(x)
        x = self.dropout1(x)
        
        # Third block with residual
        identity = x
        x = F.gelu(self.conv3a(x))
        x = self.conv3b(x)  # 1x1 conv
        x = self.bn3(x)
        x = F.gelu(x)
        if identity.size(1) != x.size(1):
            identity = F.pad(identity, (0, 0, 0, 0, 0, x.size(1) - identity.size(1)))
        x = x + identity  # Residual connection
        x = self.dropout2(x)
        
        # Classification
        x = x.view(-1, 32 * 7 * 7)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)