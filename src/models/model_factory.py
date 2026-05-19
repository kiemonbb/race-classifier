from .vgg16 import build_vgg16


def get_model(model_name, input_shape):
    if model_name == "vgg16":
        return build_vgg16(input_shape)
    else:
        raise ValueError("Unknown name")
