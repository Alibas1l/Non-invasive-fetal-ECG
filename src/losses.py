"""Combined training loss and the real validation metric.

Waveform MSE alone is a poor proxy for the thing that actually matters
clinically (fetal heart rate / beat timing): a model can get low MSE by
predicting a flat near-zero line when beats are a small fraction of the
window, while completely missing beats. So training uses a combined loss,
and validation additionally reports precision/recall/F1 of detected QRS
peaks against the real annotations within a physiological tolerance window,
independent of the loss landscape.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.signal import find_peaks
from torch import nn


class FetalECGLoss(nn.Module):
    def __init__(self, wave_weight: float = 1.0, qrs_weight: float = 2.0):
        super().__init__()
        self.wave_weight = wave_weight
        self.qrs_weight = qrs_weight
        self.wave_loss = nn.L1Loss()
        self.qrs_loss = nn.BCELoss()

    def forward(self, pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        wave_l = self.wave_loss(pred["wave"], target["y_wave"])
        qrs_l = self.qrs_loss(pred["qrs"].clamp(1e-6, 1 - 1e-6), target["y_qrs"])
        total = self.wave_weight * wave_l + self.qrs_weight * qrs_l
        return {"total": total, "wave": wave_l.detach(), "qrs": qrs_l.detach()}


def peaks_from_qrs_signal(qrs_signal: np.ndarray, fs: float, height: float = 0.5, min_rr_ms: float = 250.0) -> np.ndarray:
    """Pick discrete beat locations from the dense QRS-detection output."""
    distance = max(1, int(min_rr_ms / 1000.0 * fs))
    peaks, _ = find_peaks(qrs_signal, height=height, distance=distance)
    return peaks


def match_peaks_f1(pred_peaks: np.ndarray, true_peaks: np.ndarray, fs: float, tolerance_ms: float = 50.0) -> dict[str, float]:
    """Greedy nearest-neighbor matching within a physiological tolerance window
    (fetal QRS complexes are ~40-50ms wide), then precision/recall/F1."""
    tol = tolerance_ms / 1000.0 * fs
    if len(true_peaks) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": len(pred_peaks), "fn": 0}
    if len(pred_peaks) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": len(true_peaks)}

    used_true = np.zeros(len(true_peaks), dtype=bool)
    tp = 0
    for p in np.sort(pred_peaks):
        diffs = np.abs(true_peaks - p).astype(np.float64)
        diffs[used_true] = np.inf
        j = np.argmin(diffs)
        if diffs[j] <= tol:
            used_true[j] = True
            tp += 1

    fp = len(pred_peaks) - tp
    fn = len(true_peaks) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def qrs_detection_f1(pred_qrs: np.ndarray, true_qrs_signal: np.ndarray, fs: float, height: float = 0.5) -> dict[str, float]:
    """End-to-end metric: peak-pick both the model's QRS output and the dense
    ground-truth target, then F1-match. Operates on a single window/record's
    1D arrays (call per-sample and average, or concatenate across a split)."""
    pred_peaks = peaks_from_qrs_signal(pred_qrs, fs, height=height)
    true_peaks = peaks_from_qrs_signal(true_qrs_signal, fs, height=0.5)
    return match_peaks_f1(pred_peaks, true_peaks, fs)
