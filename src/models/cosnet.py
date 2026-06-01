from .model import Model
from torch import optim
from torchsummary import summary
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchvision import models


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 7)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.classifier(x)
        return x


class CosNet(Model):
    def __init__(self, epochs):
        model = CNNModel()

        self._model = model
        self._optimizer = optim.AdamW(
            model.classifier.parameters(), lr=1e-4, weight_decay=1e-4)
        self._scheduler = CosineAnnealingWarmRestarts(
            self._optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )

    @property
    def model(self):
        return self._model

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def scheduler(self):
        return self._scheduler

    def unfreeze_last_block(self):
        pass

    def unfreeze_penultimate_block(self):
        pass

    def unfreeze_early_block(self):
        pass
