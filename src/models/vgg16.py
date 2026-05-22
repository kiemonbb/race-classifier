import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models
from torch import optim
from .model import Model


class VGG16(Model):
    def __init__(self, epochs):
        model = models.vgg16(weights="DEFAULT")
        for param in model.parameters():
            param.requires_grad = False

        model.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        model.classifier = nn.Sequential(nn.Flatten(),
                                         nn.Dropout(0.4),
                                         nn.Linear(8192, 512),
                                         nn.BatchNorm1d(512),
                                         nn.Linear(512, 128),
                                         nn.ReLU(),
                                         nn.Dropout(0.3),
                                         nn.Linear(128, 7)
                                         )
        self._model = model
        self._optimizer = optim.Adam(
            model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
        self._scheduler = CosineAnnealingLR(
            self._optimizer, T_max=epochs, eta_min=1e-6)

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
        for param in self._model.features[-7:].parameters():
            param.requires_grad = True

        param_group = {
            'params': self._model.features[-7:].parameters(),
            'lr': 1e-5,
            'weight_decay': 1e-4
        }

        self._optimizer.add_param_group(param_group)
        self.scheduler.base_lrs.append(param_group['lr'])

    def unfreeze_penultimate_block(self):
        for param in self._model.features[-14:-7].parameters():
            param.requires_grad = True

        param_group = {
            'params': self._model.features[-14:-7].parameters(),
            'lr': 1e-6,
            'weight_decay': 1e-5
        }

        self._optimizer.add_param_group(param_group)
        self.scheduler.base_lrs.append(param_group['lr'])

    def unfreeze_early_block(self):
        for param in self._model.features[-21:-14].parameters():
            param.requires_grad = True
        param_group = {
            'params': self._model.features[-21:-14].parameters(),
            'lr': 1e-7,
            'weight_decay': 1e-5
        }
        self._optimizer.add_param_group(param_group)
        self.scheduler.base_lrs.append(param_group['lr'])
