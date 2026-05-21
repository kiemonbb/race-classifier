import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, csv_path, images_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.transform = transform
        self.classes = sorted(self.df["race"].unique().tolist())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["file"]
        label = self.class_to_idx[self.df.iloc[idx]["race"]]
        label = torch.tensor(label)
        img = Image.open(os.path.join(
            self.images_dir, img_path)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


def get_loaders(image_size=(224, 224), batch_size=32):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_csv_path = os.path.abspath(os.path.join(
        current_dir, "..", "data", "train_labels.csv"))
    val_csv_path = os.path.abspath(os.path.join(
        current_dir, "..", "data", "val_labels.csv"))
    images_dir = os.path.abspath(os.path.join(
        current_dir, "..", "data"))

    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = ImageDataset(train_csv_path, images_dir, train_transform)
    val_dataset = ImageDataset(val_csv_path, images_dir, val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=4,  shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=4,  shuffle=False)

    return train_loader, val_loader
