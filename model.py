"""
Modelldefinition: ResNet50 für Emotionsklassifikation.
"""

import torch.nn as nn
from torchvision import models


class EmotionResNet(nn.Module):
    def __init__(self, num_classes):
        super(EmotionResNet, self).__init__()
        self.resnet = models.resnet50(weights=None)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)
