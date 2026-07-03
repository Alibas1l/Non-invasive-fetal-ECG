"""Training loop for FetalECGUNet.

Validation reports the combined loss plus QRS-detection F1 against the real
annotations (via the synthetic QRS target's peaks as ground truth, matched
within a tolerance window) -- the metric that actually reflects whether
fetal heart rate is being extracted correctly, since waveform loss alone
can look good while missing beats.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import FetalECGWindowDataset, load_splits
from losses import FetalECGLoss, match_peaks_f1, peaks_from_qrs_signal
from model import FetalECGUNet

FS = 1000.0


def run_epoch(model, loader, loss_fn, optimizer=None, device="cpu") -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    totals = {"total": 0.0, "wave": 0.0, "qrs": 0.0}
    n_batches = 0
    all_f1 = []

    for batch in loader:
        x = batch["x"].to(device)
        target = {"y_qrs": batch["y_qrs"].to(device), "y_wave": batch["y_wave"].to(device)}

        with torch.set_grad_enabled(is_train):
            pred = model(x)
            losses = loss_fn(pred, target)

        if is_train:
            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()

        for k in totals:
            totals[k] += float(losses[k].detach())
        n_batches += 1

        if not is_train:
            pred_qrs = pred["qrs"].detach().cpu().numpy()
            true_qrs = batch["y_qrs"].numpy()
            for i in range(pred_qrs.shape[0]):
                pred_peaks = peaks_from_qrs_signal(pred_qrs[i], FS, height=0.5)
                true_peaks = peaks_from_qrs_signal(true_qrs[i], FS, height=0.5)
                all_f1.append(match_peaks_f1(pred_peaks, true_peaks, FS)["f1"])

    avg = {k: v / n_batches for k, v in totals.items()}
    if all_f1:
        avg["qrs_f1"] = float(np.mean(all_f1))
    return avg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-records", type=int, default=6, help="subset size for a fast smoke run; 0 = all")
    parser.add_argument("--val-records", type=int, default=3, help="subset size for a fast smoke run; 0 = all")
    parser.add_argument("--window-len", type=int, default=2000)
    parser.add_argument("--stride", type=int, default=1000)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    splits = load_splits()
    train_records = splits["train"] if args.train_records == 0 else splits["train"][: args.train_records]
    val_records = splits["val"] if args.val_records == 0 else splits["val"][: args.val_records]
    print(f"train records={len(train_records)}, val records={len(val_records)}")

    t0 = time.time()
    train_ds = FetalECGWindowDataset(train_records, window_len=args.window_len, stride=args.stride)
    val_ds = FetalECGWindowDataset(val_records, window_len=args.window_len, stride=args.stride)
    print(f"train windows={len(train_ds)}, val windows={len(val_ds)} (loaded in {time.time()-t0:.1f}s)")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = FetalECGUNet(in_channels=3, base_channels=16, depth=4).to(device)
    loss_fn = FetalECGLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = run_epoch(model, train_loader, loss_fn, optimizer, device)
        val_metrics = run_epoch(model, val_loader, loss_fn, None, device)
        dt = time.time() - t0
        print(
            f"epoch {epoch}/{args.epochs} ({dt:.1f}s) "
            f"train: total={train_metrics['total']:.4f} wave={train_metrics['wave']:.4f} qrs={train_metrics['qrs']:.4f} | "
            f"val: total={val_metrics['total']:.4f} wave={val_metrics['wave']:.4f} qrs={val_metrics['qrs']:.4f} "
            f"f1={val_metrics.get('qrs_f1', float('nan')):.3f}"
        )

    ckpt_path = "data/fetal_ecg_unet.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
