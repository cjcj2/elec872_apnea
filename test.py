import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import ApneaDetectionModel, ApneaDetectionModelNoAttention, ApneaDetectionModelFewerBands, ApneaDetectionModelSingleChannel
from train import ApneaDataset, split_subjects

@torch.no_grad()
def collect_predictions(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    probs_list = []
    labels_list = []

    for x, stage, y in tqdm(loader, desc="Evaluating", leave=False):
        x = x.to(device, non_blocking=True)
        stage = stage.to(device, non_blocking=True)

        logits = model(x, stage)
        probs = torch.sigmoid(logits).squeeze()

        probs_list.append(probs.cpu())
        labels_list.append(y.cpu())

    probs = torch.cat(probs_list).numpy()
    labels = torch.cat(labels_list).numpy()

    return probs, labels


def calculate_metrics(y_true, y_pred):
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return accuracy, precision, recall, f1, np.array([[tn, fp], [fn, tp]])


def find_optimal_threshold(y_true, probs, n_thresholds=500):
    thresholds = np.linspace(0, 1, n_thresholds)
    best_f1 = -1
    best_threshold = 0.5
    best_metrics = None

    for threshold in thresholds:
        y_pred = (probs >= threshold).astype(int)
        acc, prec, rec, f1, cm = calculate_metrics(y_true, y_pred)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = (acc, prec, rec, f1, cm)

    return best_threshold, best_metrics


def calculate_roc_auc(y_true, probs, n_points=500):
    thresholds = np.linspace(0, 1, n_points)

    n_pos = (y_true == 1).sum()
    n_neg = (y_true == 0).sum()

    tpr_list = []
    fpr_list = []

    for threshold in thresholds:
        y_pred = (probs >= threshold).astype(int)

        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()

        tpr = tp / n_pos if n_pos > 0 else 0
        fpr = fp / n_neg if n_neg > 0 else 0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    fpr = np.array(fpr_list)
    tpr = np.array(tpr_list)

    sort_idx = np.argsort(fpr)
    fpr = fpr[sort_idx]
    tpr = tpr[sort_idx]

    auc = np.trapezoid(tpr, fpr)

    return fpr, tpr, auc


def plot_results(cm, fpr, tpr, auc, ablation_type):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # confusion matrix
    ax = axes[0]
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['No Apnea', 'Apnea'])
    ax.set_yticklabels(['No Apnea', 'Apnea'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix ({ablation_type})')

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=14)

    # roc curve
    ax = axes[1]
    ax.plot(fpr, tpr, linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve (AUC = {auc:.3f})')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    # ablation selection
    ablation_type = "single_channel"

    data_dir = Path("/content/drive/MyDrive/ColabNotebooks/elec872/data/mesa_data_v4_preprocessed")

    ckpt_name = f"best_weights_{ablation_type}.pth"
    ckpt_path = Path("/content/drive/MyDrive/ColabNotebooks/elec872/outputs") / ckpt_name

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # data
    files = sorted(data_dir.glob("mesa_*_preprocessed.npz"))
    train_files, val_files, test_files = split_subjects(files, seed=42)

    # ablation-specific dataset parameters
    dataset_n_bands = 3 if ablation_type == "fewer_bands" else None
    dataset_n_channels = 1 if ablation_type == "single_channel" else None

    val_ds = ApneaDataset(val_files, n_bands=dataset_n_bands, n_channels=dataset_n_channels)
    test_ds = ApneaDataset(test_files, n_bands=dataset_n_bands, n_channels=dataset_n_channels)

    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    # load checkpoint to get model configuration
    ckpt = torch.load(ckpt_path, map_location=device)
    n_bands = ckpt["n_bands"]
    n_channels = ckpt["n_channels"]
    n_stages = ckpt["n_stages"]

    print(f"Ablation: {ablation_type}")
    print(f"Model config: n_bands={n_bands}, n_channels={n_channels}, n_stages={n_stages}")

    # select model based on ablation type
    if ablation_type == "fewer_bands":
        model = ApneaDetectionModelFewerBands(
            n_bands=n_bands,
            n_channels=n_channels,
            n_stages=n_stages,
            d_model=128,
            n_heads=4,
            dropout=0.2,
            attn_dropout=0.1,
        ).to(device)
    elif ablation_type == "no_attention":
        model = ApneaDetectionModelNoAttention(
            n_bands=n_bands,
            n_channels=n_channels,
            n_stages=n_stages,
            d_model=128,
            dropout=0.2,
        ).to(device)
    elif ablation_type == "single_channel":
        model = ApneaDetectionModelSingleChannel(
            n_bands=n_bands,
            n_channels=n_channels,
            n_stages=n_stages,
            d_model=128,
            n_heads=4,
            dropout=0.2,
            attn_dropout=0.1,
        ).to(device)
    else:  # baseline
        model = ApneaDetectionModel(
            n_bands=n_bands,
            n_channels=n_channels,
            n_stages=n_stages,
            d_model=128,
            n_heads=4,
            dropout=0.2,
            attn_dropout=0.1,
        ).to(device)

    model.load_state_dict(ckpt["model_state"])

    # optimal threshold
    val_probs, val_labels = collect_predictions(model, val_loader, device)
    best_threshold, (val_acc, val_prec, val_rec, val_f1, val_cm) = find_optimal_threshold(val_labels, val_probs)

    print(f"\nValidation Results (Threshold = {best_threshold:.3f}):")
    print(f"Accuracy:  {val_acc:.3f}")
    print(f"Precision: {val_prec:.3f}")
    print(f"Recall:    {val_rec:.3f}")
    print(f"F1 Score:  {val_f1:.3f}")

    # test metrics
    test_probs, test_labels = collect_predictions(model, test_loader, device)
    test_pred = (test_probs >= best_threshold).astype(int)
    test_acc, test_prec, test_rec, test_f1, test_cm = calculate_metrics(test_labels, test_pred)
    fpr, tpr, auc = calculate_roc_auc(test_labels, test_probs)

    print(f"\nTest Results (Threshold = {best_threshold:.3f}):")
    print(f"Accuracy:  {test_acc:.3f}")
    print(f"Precision: {test_prec:.3f}")
    print(f"Recall:    {test_rec:.3f}")
    print(f"F1 Score:  {test_f1:.3f}")
    print(f"ROC-AUC:   {auc:.3f}")

    # results
    plot_results(test_cm, fpr, tpr, auc, ablation_type)


if __name__ == "__main__":
    main()