# Here’s a simplified version of your CNN_TUMOR clas

import torch.nn as nn
import torch.nn.functional as F

class CNN_TUMOR(nn.Module):
    def __init__(self, shape_in=(3, 256, 256), initial_filters=8, num_fc1=100, num_classes=2, dropout_rate=0.5):
        super(CNN_TUMOR, self).__init__()
        self.conv1 = nn.Conv2d(3, initial_filters, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(initial_filters, initial_filters * 2, kernel_size=3, padding=1)

        # Dynamically compute flatten size
        dummy = torch.zeros(1, *shape_in)
        x = self.forward_features(dummy)
        self.flatten_size = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(self.flatten_size, num_fc1)
        self.fc2 = nn.Linear(num_fc1, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward_features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x.view(-1, self.flatten_size)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

