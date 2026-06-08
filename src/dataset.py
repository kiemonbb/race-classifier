import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import transforms
import os
from math import ceil, floor
from PIL import Image


class TransformSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


class ImageDataset(Dataset):
    def __init__(self, csv_path, images_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.transform = transform
        self.classes = sorted(self.df["race"].unique().tolist())
        # self.classes = ["Black", "White", "Other"]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        # self.class_to_idx = {'Black': 0, 'East Asian': 2, 'Indian': 2, 'Latino_Hispanic': 2, 'Middle Eastern': 2, 'Southeast Asian': 2, 'White': 1}

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
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomGrayscale(p=0.05),
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

    train_test_dataset = ImageDataset(
        train_csv_path, images_dir)

    indices = list(range(len(train_test_dataset)))
    labels = train_test_dataset.df["race"].tolist()

    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.17,
        random_state=42,
        stratify=labels
    )
    train_dataset = TransformSubset(
        Subset(train_test_dataset, train_idx), train_transform)
    test_dataset = TransformSubset(
        Subset(train_test_dataset, test_idx), val_transform)
    val_dataset = ImageDataset(val_csv_path, images_dir, val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=4,  shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=4,  shuffle=False)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=4,  shuffle=False)

    return train_loader, val_loader, test_loader
