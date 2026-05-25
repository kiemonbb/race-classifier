from .vgg16 import VGG16
from .resnet50 import ResNet50


def get_model(model_name, epochs):
    if model_name == "vgg16":
        return VGG16(epochs)
    elif model_name == "resnet50":
        return ResNet50(epochs)
    else:
        raise ValueError("Unknown name")
