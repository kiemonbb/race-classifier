import matplotlib.pyplot as plt
import argparse
import yaml
from torchsummary import summary
import torch
import os
import datetime
from tqdm import tqdm
import torch.nn as nn
from torch import optim
from models.model_factory import get_model
from dataset import get_loaders


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def plot_training(history,  train_results_path):

    plt.plot(history["accuracy"])
    plt.plot(history['val_accuracy'])
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Accuracy", "Validation Accuracy"])

    acc_jpg_path = os.path.join(train_results_path, "acc.jpg")

    plt.savefig(acc_jpg_path)
    plt.show()

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Loss", "Validation Loss"])

    loss_jpg_path = os.path.join(train_results_path, "loss.jpg")

    plt.savefig(loss_jpg_path)
    plt.show()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_name = config["model_name"]
    image_size = tuple(config["image_size"])
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    steps_per_epoch = config["steps_per_epoch"]
    steps_per_val = config["steps_per_val"]
    learning_rate = config["learning_rate"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(
        model_name=model_name,
    ).to(device)
    summary(model, input_size=(3, 224, 224))
    print(f"Device: {device}")

    train_loader, val_loader = get_loaders(image_size, batch_size)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    best_val_accuracy = 0.0

    history = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{model_name}_{timestamp}"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_results_path = os.path.abspath(os.path.join(
        current_dir, "..", "results", "train", run_name))
    os.makedirs(train_results_path, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        correct, total, running_loss = 0, 0, 0.0
        train_steps = steps_per_epoch if steps_per_epoch else len(train_loader)
        bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [TRAIN]",
                   total=train_steps)
        for i, (inputs, labels) in enumerate(train_loader):
            if steps_per_epoch and i >= steps_per_epoch:
                break
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
            bar.set_postfix(
                loss=f"{running_loss/(i+1):.4f}", acc=f"{correct/total:.4f}", refresh=True)
            bar.update(1)
        history["loss"].append(
            running_loss / train_steps)
        history["accuracy"].append(correct/total)

        model.eval()
        correct, total, running_loss = 0, 0, 0.0
        val_steps = steps_per_val if steps_per_val else len(val_loader)
        bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [VALID]",
                   total=val_steps)
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_loader):
                if steps_per_val and i >= steps_per_val:
                    break
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)
                bar.set_postfix(
                    loss=f"{running_loss/(i+1):.4f}", acc=f"{correct/total:.4f}", refresh=True)
                bar.update(1)
        history["val_loss"].append(
            running_loss / val_steps)
        history["val_accuracy"].append(correct / total)

        # CHECKPOINT
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history": history,
        }, os.path.join(train_results_path, "checkpoint.pth"))

        # BEST MODEL WEIGHTS
        if history["val_accuracy"][-1] > best_val_accuracy:
            best_val_accuracy = history["val_accuracy"][-1]
            torch.save(model.state_dict(), os.path.join(
                train_results_path, "model.pth"))

    plot_training(history, train_results_path)


if __name__ == "__main__":
    main()
