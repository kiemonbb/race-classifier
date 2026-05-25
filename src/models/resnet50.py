from .model import Model
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights
from torch.optim.lr_scheduler import OneCycleLR
from torch import optim


class ResNet50(Model):
    def __init__(self, epochs):
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        for param in model.parameters():
            param.requires_grad = False

        classifier_layers = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.4),
                                 nn.Linear(classifier_layers, 512),
                                 nn.BatchNorm1d(512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 7)
                                 )
        self._model = model

        self._optimizer = optim.AdamW(
            model.fc.parameters(), lr=1e-3, weight_decay=3e-4)
        self._scheduler = OneCycleLR(
            self._optimizer, max_lr=1e-2, epochs=3, steps_per_epoch=2800)

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
        for param in self._model.layer4.parameters():
            param.requires_grad = True

        param_group = {
            'params': self._model.layer4.parameters(),
            'lr': 5e-5,
            'weight_decay': 1e-4,
        }

        self._optimizer.add_param_group(param_group)
        self._scheduler = OneCycleLR(
            self._optimizer,
            max_lr=[1e-3, 1e-4],
            epochs=3,
            steps_per_epoch=2800,
        )

    def unfreeze_penultimate_block(self):
        for param in self._model.layer3.parameters():
            param.requires_grad = True

        param_group = {
            'params': self._model.layer3.parameters(),
            'lr': 1e-5,
            'weight_decay': 1e-4,
        }

        self._optimizer.add_param_group(param_group)
        self._scheduler = OneCycleLR(
            self._optimizer,
            max_lr=[1e-3, 1e-4, 5e-5],
            epochs=19,
            steps_per_epoch=2800,
        )
        pass

    def unfreeze_early_block(self):
        pass
