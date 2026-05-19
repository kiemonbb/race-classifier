from .vgg16 import build_vgg16


def get_model(model_name):
    if model_name == "vgg16":
        return build_vgg16()
    else:
        raise ValueError("Unknown name")
