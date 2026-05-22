import matplotlib.pyplot as plt
from torch.nn.functional import softmax
import argparse
import yaml
from torchsummary import summary
import torch
import os
import datetime
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
from torch import optim
from models.model_factory import get_model
from dataset import get_loaders


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-w", "--weights", required=True)
    parser.add_argument("-i", "--img", required=True)
    return parser.parse_args()


def use_model():
    args = parse_args()
    current_dir = os.path.dirname(os.path.abspath(__file__))

    weights_path = os.path.abspath(os.path.join(current_dir, args.weights))
    image_path = os.path.abspath(os.path.join(current_dir, args.img))
    img = Image.open(image_path).convert("RGB")

    model, optimizer = get_model(str(args.model), 0)
    state_dict = torch.load(weights_path)
    model.load_state_dict(state_dict)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    input_img = transform(img)
    input_batch = input_img.unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input_batch = input_batch.to(device)

    with torch.no_grad():
        output = model(input_batch)
    probablilites = softmax(output, dim=1).tolist()
    labels = ['Black', 'East Asian', 'Indian', 'Latino_Hispanic',
              'Middle Eastern', 'Southeast Asian', 'White']
    for i in range(7):
        print(f"{labels[i]}: {probablilites[0][i]*100:.2f}%")


if __name__ == "__main__":
    use_model()
