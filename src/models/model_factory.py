from .resnet50 import build_resnet50
from .vgg16 import VGG16


def get_model(model_name, epochs):
    if model_name == "vgg16":
        return VGG16(epochs)
    elif model_name == "resnet50":
        return build_resnet50()
    else:
        raise ValueError("Unknown name")
