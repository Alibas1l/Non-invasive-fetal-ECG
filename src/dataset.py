"""PyTorch Dataset wrapping data_pipeline: loads each record once, windows it
into (X, y_qrs, y_wave) triples in memory. The dataset holds one split
(train/val/test) of records at a time -- with 55 records of a few hundred
seconds each at 1kHz this fits comfortably in memory, so there's no need for
lazy per-window disk access."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data_pipeline import ROOT, load_record, window_record

SPLITS_PATH = ROOT / "data" / "splits.json"


class FetalECGWindowDataset(Dataset):
    def __init__(self, records: list[str], window_len: int = 2000, stride: int = 1000):
        self.window_len = window_len
        self.stride = stride

        X_parts, y_qrs_parts, y_wave_parts = [], [], []
        for record in records:
            rec = load_record(record)
            X, y_qrs, y_wave = window_record(rec, window_len=window_len, stride=stride)
            X_parts.append(X)
            y_qrs_parts.append(y_qrs)
            y_wave_parts.append(y_wave)

        self.X = np.concatenate(X_parts, axis=0)
        self.y_qrs = np.concatenate(y_qrs_parts, axis=0)
        self.y_wave = np.concatenate(y_wave_parts, axis=0)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "x": torch.from_numpy(self.X[idx]),
            "y_qrs": torch.from_numpy(self.y_qrs[idx]),
            "y_wave": torch.from_numpy(self.y_wave[idx]),
        }


def load_splits() -> dict[str, list[str]]:
    return json.loads(SPLITS_PATH.read_text())


if __name__ == "__main__":
    splits = load_splits()
    # Small smoke test: 3 records from train, all of val (both small enough to be fast)
    ds = FetalECGWindowDataset(splits["train"][:3], window_len=2000, stride=1000)
    print(f"train subset windows: {len(ds)}")
    sample = ds[0]
    print({k: tuple(v.shape) for k, v in sample.items()})
