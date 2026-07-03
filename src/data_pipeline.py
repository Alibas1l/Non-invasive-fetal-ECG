"""Core data pipeline: load NIFECGDB records, window them, and produce
subject-level train/val/test splits for a 1D-CNN fetal-ECG extractor.

There is no isolated "clean FECG" channel in this dataset -- only maternal
thorax leads and a noisy abdominal composite, plus fetal QRS *locations*
(not waveforms) in the .qrs annotations. Ground truth is therefore built,
not recorded, from those locations two ways:

  1. `qrs_target_signal` -- a narrow Gaussian bump at each annotated beat,
     for training/evaluating a fetal-QRS *detector*.
  2. `synthesize_fetal_ecg` -- a McSharry-style sum-of-Gaussians PQRST
     template anchored to the real (accurate) QRS timing, with each beat's
     width scaled to the local RR interval. This gives a physiologically
     shaped, amplitude-normalized "clean FECG" waveform to regress against
     for a true waveform-extraction/denoising CNN, without needing a
     separate synthetic noisy-signal generator -- the real noisy abdominal
     recording is used as-is for the input side.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import wfdb

mne.set_log_level("ERROR")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "non-invasive-fetal-ecg-database-1.0.0" / "non-invasive-fetal-ecg-database-1.0.0"
MANIFEST_PATH = ROOT / "data" / "manifest.json"

N_ABDOMEN_CHANNELS = 3  # minimum common count across all records
FS = 1000.0


def list_records() -> list[str]:
    records_file = DATA_DIR / "RECORDS"
    return [line.strip().removesuffix(".edf") for line in records_file.read_text().splitlines() if line.strip()]


@dataclass
class RecordData:
    record: str
    abdomen: np.ndarray  # (N_ABDOMEN_CHANNELS, T), float32
    qrs_samples: np.ndarray  # int sample indices of fetal QRS annotations
    fs: float


def load_record(record: str) -> RecordData:
    raw = mne.io.read_raw_edf(str(DATA_DIR / f"{record}.edf"), preload=True, verbose="ERROR")
    labels = raw.ch_names
    abdomen_idx = [i for i, name in enumerate(labels) if name.startswith("Abdomen")][:N_ABDOMEN_CHANNELS]
    data = raw.get_data(picks=abdomen_idx).astype(np.float32)  # volts

    ann = wfdb.rdann(str(DATA_DIR / record), extension="edf.qrs")
    qrs_samples = np.asarray(ann.sample, dtype=np.int64)

    return RecordData(record=record, abdomen=data, qrs_samples=qrs_samples, fs=raw.info["sfreq"])


def qrs_target_signal(qrs_samples: np.ndarray, n_samples: int, fs: float, sigma_ms: float = 15.0) -> np.ndarray:
    """Dense target: sum of narrow Gaussians centered at each fetal QRS sample.
    Used as the regression/detection target for the CNN instead of a clean waveform."""
    target = np.zeros(n_samples, dtype=np.float32)
    sigma = sigma_ms / 1000.0 * fs
    half_width = int(6 * sigma)
    for s in qrs_samples:
        lo, hi = max(0, s - half_width), min(n_samples, s + half_width)
        idx = np.arange(lo, hi) - s
        target[lo:hi] = np.maximum(target[lo:hi], np.exp(-0.5 * (idx / sigma) ** 2))
    return target


# McSharry et al. (2003) ECGSYN dynamical-model PQRST parameters: each wave is a
# Gaussian on a beat-cycle angle theta in (-pi, pi], theta=0 at the R peak.
# Amplitudes are relative; the template is renormalized so the R peak is exactly 1.
_WAVE_THETA = {"P": -np.pi / 3, "Q": -np.pi / 12, "R": 0.0, "S": np.pi / 12, "T": np.pi / 2}
_WAVE_A = {"P": 1.2, "Q": -5.0, "R": 30.0, "S": -7.5, "T": 0.75}
_WAVE_B = {"P": 0.25, "Q": 0.1, "R": 0.1, "S": 0.1, "T": 0.4}


def _pqrst_template(theta: np.ndarray) -> np.ndarray:
    """Evaluate the sum-of-Gaussians PQRST shape at beat-cycle angles theta (radians)."""
    wrapped = (theta + np.pi) % (2 * np.pi) - np.pi
    y = np.zeros_like(theta)
    for wave, theta_wave in _WAVE_THETA.items():
        d = wrapped - theta_wave
        d = (d + np.pi) % (2 * np.pi) - np.pi  # shortest angular distance
        y += _WAVE_A[wave] * np.exp(-0.5 * (d / _WAVE_B[wave]) ** 2)
    return y / _WAVE_A["R"]


def synthesize_fetal_ecg(qrs_samples: np.ndarray, n_samples: int, fs: float) -> np.ndarray:
    """Build a clean, amplitude-normalized synthetic FECG waveform anchored to the
    real fetal QRS sample locations. Each beat's PQRST template is stretched to
    match the local RR interval (half from the previous beat, half to the next),
    so beat width tracks the true instantaneous fetal heart rate."""
    target = np.zeros(n_samples, dtype=np.float32)
    qrs_samples = np.sort(qrs_samples)
    if len(qrs_samples) == 0:
        return target

    for i, s in enumerate(qrs_samples):
        prev_rr = s - qrs_samples[i - 1] if i > 0 else (qrs_samples[i + 1] - s if len(qrs_samples) > 1 else fs)
        next_rr = qrs_samples[i + 1] - s if i < len(qrs_samples) - 1 else prev_rr
        lo, hi = max(0, s - prev_rr // 2), min(n_samples, s + next_rr // 2)
        if lo >= hi:
            continue
        idx = np.arange(lo, hi) - s
        half_span = np.where(idx < 0, prev_rr / 2, next_rr / 2)
        theta = idx / half_span * np.pi
        # Per-beat windows are contiguous (tile lo..hi across the whole record), so a
        # direct assignment is correct here -- np.maximum would clip the negative Q/S
        # dips to the zero baseline instead of overwriting it.
        target[lo:hi] = _pqrst_template(theta).astype(np.float32)
    return target


def zscore(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    mean = x.mean(axis=axis, keepdims=True)
    std = x.std(axis=axis, keepdims=True)
    return (x - mean) / (std + eps)


def window_record(
    rec: RecordData, window_len: int = 2000, stride: int = 1000
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slice a record into overlapping windows.
    Returns (X, y_qrs, y_wave):
      X      shape (n_windows, N_ABDOMEN_CHANNELS, window_len) -- real, z-scored input
      y_qrs  shape (n_windows, window_len) -- Gaussian-bump fetal-QRS detection target
      y_wave shape (n_windows, window_len) -- synthetic clean-FECG waveform target"""
    n_samples = rec.abdomen.shape[1]
    qrs_target = qrs_target_signal(rec.qrs_samples, n_samples, rec.fs)
    wave_target = synthesize_fetal_ecg(rec.qrs_samples, n_samples, rec.fs)

    starts = list(range(0, n_samples - window_len + 1, stride))
    X = np.empty((len(starts), rec.abdomen.shape[0], window_len), dtype=np.float32)
    y_qrs = np.empty((len(starts), window_len), dtype=np.float32)
    y_wave = np.empty((len(starts), window_len), dtype=np.float32)
    for i, s in enumerate(starts):
        X[i] = zscore(rec.abdomen[:, s : s + window_len], axis=-1)
        y_qrs[i] = qrs_target[s : s + window_len]
        y_wave[i] = wave_target[s : s + window_len]
    return X, y_qrs, y_wave


def split_records(records: list[str], seed: int = 42, train_frac: float = 0.7, val_frac: float = 0.15):
    rng = np.random.default_rng(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    n_train = int(len(shuffled) * train_frac)
    n_val = int(len(shuffled) * val_frac)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def build_split_manifest(seed: int = 42) -> dict:
    records = list_records()
    splits = split_records(records, seed=seed)
    out_path = ROOT / "data" / "splits.json"
    out_path.write_text(json.dumps(splits, indent=2))
    return splits


if __name__ == "__main__":
    splits = build_split_manifest()
    print({k: len(v) for k, v in splits.items()})

    # Smoke test on one record from each split
    for name, recs in splits.items():
        rec = load_record(recs[0])
        X, y_qrs, y_wave = window_record(rec)
        print(
            f"[{name}] {rec.record}: abdomen shape={rec.abdomen.shape}, "
            f"n_qrs={len(rec.qrs_samples)}, windows X={X.shape} "
            f"y_qrs={y_qrs.shape} (max={y_qrs.max():.3f}) "
            f"y_wave={y_wave.shape} (max={y_wave.max():.3f}, min={y_wave.min():.3f})"
        )
