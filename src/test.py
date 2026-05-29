import argparse
import torch
import os
from tqdm import tqdm
import pandas as pd
import datetime
from models.model_factory import get_model
from dataset import get_loaders
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-w", "--weights", required=True)
    return parser.parse_args()


def test_model():
    args = parse_args()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.abspath(os.path.join(current_dir, args.weights))

    model_name = str(args.model)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{model_name}_{timestamp}"
    test_results_path = os.path.abspath(os.path.join(
        current_dir, "..", "results", "test", run_name))

    os.makedirs(test_results_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_structure = get_model(model_name, 0)
    state_dict = torch.load(weights_path, map_location=device)
    model_structure.model.load_state_dict(state_dict,)

    model_structure.model.eval()

    train_loader, val_loader, test_loader = get_loaders((224, 224), 32)
    class_names = train_loader.dataset.subset.dataset.classes
    correct, total = 0, 0
    all_preds = []
    all_labels = []

    correct, total = 0, 0
    test_steps = len(test_loader)
    bar = tqdm(test_loader, desc=f"[TEST]",
               total=test_steps)
    with torch.no_grad():
        for i, data in enumerate(bar):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model_structure.model(inputs)

            preds = outputs.argmax(1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
            bar.set_postfix(
                acc=f"{correct/total:.4f}", refresh=True)
            bar.update(1)

    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    confusion_matrix_path = os.path.join(
        test_results_path, "confusion_matrix.jpg")
    plt.savefig(confusion_matrix_path)

    plt.show()

    accuracy = accuracy_score(all_labels, all_preds)
    balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro")

    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
    print(f"Macro Precision: {precision_macro:.4f}")
    print(f"Macro Recall: {recall_macro:.4f}")
    print(f"Macro F1: {f1_macro:.4f}")
    print(f"Weighted Precision: {precision_weighted:.4f}")
    print(f"Weighted Recall: {recall_weighted:.4f}")
    print(f"Weighted F1: {f1_weighted:.4f}")

    evaluation_metrics_path = os.path.join(
        test_results_path, "evaluation_metrics.txt")

    with open(evaluation_metrics_path, "x") as f:
        print(f"Accuracy: {accuracy:.4f}", file=f)
        print(f"Balanced Accuracy: {balanced_accuracy:.4f}", file=f)
        print(f"Macro Precision: {precision_macro:.4f}", file=f)
        print(f"Macro Recall: {recall_macro:.4f}", file=f)
        print(f"Macro F1: {f1_macro:.4f}", file=f)
        print(f"Weighted Precision: {precision_weighted:.4f}", file=f)
        print(f"Weighted Recall: {recall_weighted:.4f}", file=f)
        print(f"Weighted F1: {f1_weighted:.4f}", file=f)

    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        output_dict=True
    )
    report_df = pd.DataFrame(report).transpose()
    report_csv_path = os.path.join(
        test_results_path, "classification_report.csv")
    report_df.to_csv(report_csv_path)


if __name__ == "__main__":
    test_model()
