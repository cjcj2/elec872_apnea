import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import ApneaDetectionModel, ApneaDetectionModelNoAttention, ApneaDetectionModelFewerBands, ApneaDetectionModelSingleChannel

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ApneaDataset(Dataset):
    def __init__(self, files: List[Path], n_bands: Optional[int] = None, n_channels: Optional[int] = None):
        self.files = list(files)
        self.index = []
        self._cached = {}
        self.n_bands_select = n_bands  # for ablation
        self.n_channels_select = n_channels  # for ablation

        self.stage_map = {1: 0, 2: 1, 3: 2, 5: 3}
        self.n_stages = 4

        for fi, fp in enumerate(self.files):
            d = np.load(fp, allow_pickle=True)
            n_epochs = int(d["eeg_bands"].shape[0])
            for ei in range(n_epochs):
                self.index.append((fi, ei))

        # counts for imbalance handling
        pos = 0
        neg = 0
        for fp in self.files:
            d = np.load(fp, allow_pickle=True)
            a = d["apnea"].astype(np.int64)
            pos += int(a.sum())
            neg += int((a == 0).sum())
        self.pos = pos
        self.neg = neg

    def __len__(self):
        return len(self.index)

    def _load_file(self, file_idx: int):
        if file_idx in self._cached:
            return self._cached[file_idx]

        fp = self.files[file_idx]
        d = np.load(fp, allow_pickle=True)
        eeg_bands = d["eeg_bands"].astype(np.float32)
        stage = d["stage"].astype(np.int64)
        apnea = d["apnea"].astype(np.float32)

        self._cached[file_idx] = (eeg_bands, stage, apnea)
        return eeg_bands, stage, apnea

    def __getitem__(self, idx: int):
        file_idx, epoch_idx = self.index[idx]
        eeg_bands, stage, apnea = self._load_file(file_idx)

        x = torch.from_numpy(eeg_bands[epoch_idx])

        # ablation fewer bands
        if self.n_bands_select is not None:
            x = x[:self.n_bands_select, :, :]

        # ablation single channel
        if self.n_channels_select is not None:
            x = x[:, :self.n_channels_select, :]

        s_raw = int(stage[epoch_idx])
        s = torch.tensor(self.stage_map[s_raw], dtype=torch.long)

        y = torch.tensor(float(apnea[epoch_idx]), dtype=torch.float32)
        return x, s, y


@torch.no_grad()
def confusion(logits: torch.Tensor, y: torch.Tensor, thresh: float = 0.5):
    probs = torch.sigmoid(logits)
    pred = (probs >= thresh).to(torch.int64)
    y = y.to(torch.int64)

    tp = int(((pred == 1) & (y == 1)).sum().item())
    tn = int(((pred == 0) & (y == 0)).sum().item())
    fp = int(((pred == 1) & (y == 0)).sum().item())
    fn = int(((pred == 0) & (y == 1)).sum().item())
    return tp, tn, fp, fn


def f1_from_conf(tp: int, fp: int, fn: int) -> float:
    denom = (2 * tp + fp + fn)
    return (2 * tp / denom) if denom > 0 else 0.0


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: str,
    desc: str,
):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    n_seen = 0
    all_logits = []
    all_y = []

    pbar = tqdm(loader, desc=desc, leave=False)
    for x, stage, y in pbar:
        x = x.to(device, non_blocking=True)
        stage = stage.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(x, stage)
        loss = criterion(logits, y)

        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        bs = x.size(0)
        total_loss += float(loss.item()) * bs
        n_seen += bs

        all_logits.append(logits.detach().cpu())
        all_y.append(y.detach().cpu())
        pbar.set_postfix(loss=total_loss / max(n_seen, 1))

    all_logits = torch.cat(all_logits, dim=0)
    all_y = torch.cat(all_y, dim=0)

    tp, tn, fp, fn = confusion(all_logits, all_y, thresh=0.5)
    f1 = f1_from_conf(tp, fp, fn)
    avg_loss = total_loss / max(n_seen, 1)

    return {"loss": avg_loss, "f1": f1}


def split_subjects(files: List[Path], seed: int = 42, train_frac: float = 0.80, val_frac: float = 0.10):
    rng = random.Random(seed)
    files = list(files)
    rng.shuffle(files)

    n = len(files)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    return files[:n_train], files[n_train:n_train + n_val], files[n_train + n_val:]


def main():
    set_seed(42)

    # ablation selection
    ablation_type = "single_channel"

    data_dir = Path("/content/drive/MyDrive/ColabNotebooks/elec872/data/mesa_data_v4_preprocessed")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    files = sorted(data_dir.glob("mesa_*_preprocessed.npz"))

    train_files, val_files, test_files = split_subjects(files, seed=42)
    print(f"split: total={len(files)} train={len(train_files)} val={len(val_files)} test={len(test_files)}")

    # ablation-specific dataset parameters
    dataset_n_bands = 3 if ablation_type == "fewer_bands" else None
    dataset_n_channels = 1 if ablation_type == "single_channel" else None

    train_ds = ApneaDataset(train_files, n_bands=dataset_n_bands, n_channels=dataset_n_channels)
    val_ds = ApneaDataset(val_files, n_bands=dataset_n_bands, n_channels=dataset_n_channels)

    # sqrt imbalance weighting
    ratio = train_ds.neg / max(train_ds.pos, 1)
    pos_weight = float(np.sqrt(ratio))
    pos_weight_t = torch.tensor([pos_weight], dtype=torch.float32, device=device)

    print(f"class balance: pos={train_ds.pos} neg={train_ds.neg} ratio={ratio:.3f} pos_weight(sqrt)={pos_weight:.3f}")

    # get data dimensions
    d0 = np.load(train_files[0], allow_pickle=True)
    n_bands_full = int(d0["eeg_bands"].shape[1])
    n_channels_full = int(d0["eeg_bands"].shape[2])
    T = int(d0["eeg_bands"].shape[-1])
    n_stages = train_ds.n_stages

    # ablation-specific model parameters
    if ablation_type == "fewer_bands":
        n_bands = 3  # delta, theta, alpha
        n_channels = n_channels_full
    elif ablation_type == "single_channel":
        n_bands = n_bands_full
        n_channels = 1
    else:
        n_bands = n_bands_full
        n_channels = n_channels_full

    print(f"ablation: {ablation_type}")
    print(f"model input: n_bands={n_bands}, n_channels={n_channels}, T={T}, n_stages={n_stages}")

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

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
    else:
        model = ApneaDetectionModel(
            n_bands=n_bands,
            n_channels=n_channels,
            n_stages=n_stages,
            d_model=128,
            n_heads=4,
            dropout=0.2,
            attn_dropout=0.1,
        ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)

    # early stopping hyperparams
    max_epochs = 50
    patience = 6
    min_delta = 1e-3

    ckpt_name = f"best_weights_{ablation_type}.pth"
    ckpt_path = Path("/content/drive/MyDrive/ColabNotebooks/elec872/outputs") / ckpt_name
    best_val_f1 = -1.0
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        train_metrics = run_one_epoch(model, train_loader, criterion, optimizer, device, desc=f"Train {epoch}")
        val_metrics = run_one_epoch(model, val_loader, criterion, optimizer=None, device=device, desc=f"Val {epoch}")

        print(
            f"Epoch {epoch:02d} | "
            f"train loss = {train_metrics['loss']:.4f} f1 = {train_metrics['f1']:.3f} | "
            f"val loss = {val_metrics['loss']:.4f} f1 = {val_metrics['f1']:.3f}")

        improved = (val_metrics["f1"] > best_val_f1 + min_delta)
        if improved:
            best_val_f1 = val_metrics["f1"]
            epochs_no_improve = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "ablation_type": ablation_type,
                    "n_bands": n_bands,
                    "n_channels": n_channels,
                    "n_stages": n_stages,
                    "pos_weight": pos_weight,
                    "best_val_f1": best_val_f1,
                },
                ckpt_path,
            )
            print(f"\tsaved checkpoint: val f1={best_val_f1:.3f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\tearly stopping")
                break


if __name__ == "__main__":
    main()