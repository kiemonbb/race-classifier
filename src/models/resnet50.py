import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights
from torch import optim


def build_resnet50(learning_rate):
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    classifier_layers = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.5),
                             nn.Linear(classifier_layers, 512),
                             nn.BatchNorm1d(512),
                             nn.ReLU(),
                             nn.Dropout(0.3),
                             nn.Linear(512, 7)
                             )

    optimizer = optim.Adam([{'params': model.layer4.parameters(), 'lr': 1e-5},
                            {'params': model.fc.parameters(), 'lr': learning_rate}])
    return model, optimizer
