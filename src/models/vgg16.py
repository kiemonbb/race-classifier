import torch.nn as nn
from torchvision import models


def build_vgg16():
    model = models.vgg16(weights="DEFAULT")

    for param in model.features[:-4].parameters():
        param.requires_grad = False

    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    model.classifier = nn.Sequential(nn.Flatten(),
                                     nn.Dropout(0.5),
                                     nn.Linear(512, 512),
                                     nn.BatchNorm1d(512),
                                     nn.ReLU(),
                                     nn.Dropout(0.3),
                                     nn.Linear(512, 7)
                                     )
    return model
