"""Sanity-check plot: real (z-scored) abdominal signal vs. the two synthetic
targets (fetal-QRS detection bump, synthetic clean-FECG waveform) over a
short window, so the alignment can be checked visually before training."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data_pipeline import list_records, load_record, qrs_target_signal, synthesize_fetal_ecg, zscore

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "target_sanity_check.png"

# Palette: muted ink for the real signal (context), one categorical hue per
# synthetic target so identity reads without relying on a rainbow.
COLOR_SIGNAL = "#6b7280"   # neutral gray -- the real, noisy input
COLOR_QRS = "#2563eb"      # blue -- detection target
COLOR_WAVE = "#dc2626"     # red -- waveform target


def main() -> None:
    record = list_records()[0]
    rec = load_record(record)

    fs = rec.fs
    start_s, end_s = 10.0, 14.0  # a 4s window, long enough to show several fetal beats
    lo, hi = int(start_s * fs), int(end_s * fs)

    signal = zscore(rec.abdomen[:, lo:hi], axis=-1)[0]  # first abdominal channel
    qrs_target = qrs_target_signal(rec.qrs_samples, rec.abdomen.shape[1], fs)[lo:hi]
    wave_target = synthesize_fetal_ecg(rec.qrs_samples, rec.abdomen.shape[1], fs)[lo:hi]
    t = np.arange(lo, hi) / fs

    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(f"Ground-truth sanity check -- record {record}, abdominal ch. 1", fontsize=11)

    axes[0].plot(t, signal, color=COLOR_SIGNAL, linewidth=1)
    axes[0].set_ylabel("Real signal\n(z-scored)")

    axes[1].plot(t, qrs_target, color=COLOR_QRS, linewidth=1.5)
    axes[1].set_ylabel("QRS detection\ntarget")

    axes[2].plot(t, wave_target, color=COLOR_WAVE, linewidth=1.5)
    axes[2].set_ylabel("Synthetic FECG\nwaveform target")
    axes[2].set_xlabel("Time (s)")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
