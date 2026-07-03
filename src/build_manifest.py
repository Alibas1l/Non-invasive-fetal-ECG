"""Scan every record in the dataset and write a manifest (JSON) describing
channel layout, sample rate, duration, and QRS annotation counts. This lets
the pipeline validate assumptions (fixed fs, channel naming) across the full
dataset instead of a handful of samples.
"""
import json
from pathlib import Path

import mne
import wfdb

mne.set_log_level("ERROR")

DATA_DIR = Path(__file__).resolve().parent.parent / "non-invasive-fetal-ecg-database-1.0.0" / "non-invasive-fetal-ecg-database-1.0.0"
MANIFEST_PATH = Path(__file__).resolve().parent.parent / "data" / "manifest.json"


def list_records() -> list[str]:
    records_file = DATA_DIR / "RECORDS"
    return [line.strip().removesuffix(".edf") for line in records_file.read_text().splitlines() if line.strip()]


def scan_record(record: str) -> dict:
    raw = mne.io.read_raw_edf(str(DATA_DIR / f"{record}.edf"), preload=False, verbose="ERROR")
    ann = wfdb.rdann(str(DATA_DIR / record), extension="edf.qrs")
    return {
        "record": record,
        "n_channels": int(len(raw.ch_names)),
        "labels": list(raw.ch_names),
        "fs": float(raw.info["sfreq"]),
        "duration_s": float(raw.n_times / raw.info["sfreq"]),
        "n_samples": int(raw.n_times),
        "n_qrs_annotations": int(len(ann.sample)),
        "qrs_symbols": sorted(str(s) for s in set(ann.symbol)),
    }


def main() -> None:
    records = list_records()
    manifest = []
    errors = []
    for record in records:
        try:
            manifest.append(scan_record(record))
        except Exception as e:
            errors.append({"record": record, "error": str(e)})

    MANIFEST_PATH.parent.mkdir(exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps({"records": manifest, "errors": errors}, indent=2))

    fs_values = {r["fs"] for r in manifest}
    channel_counts = {r["n_channels"] for r in manifest}
    label_layouts = {tuple(r["labels"]) for r in manifest}
    durations = [r["duration_s"] for r in manifest]

    print(f"Scanned {len(manifest)}/{len(records)} records ({len(errors)} errors)")
    print(f"Sample rates seen: {fs_values}")
    print(f"Channel counts seen: {channel_counts}")
    print(f"Distinct label layouts: {len(label_layouts)}")
    for layout in label_layouts:
        print(f"  {layout}")
    print(f"Duration range: {min(durations):.1f}s - {max(durations):.1f}s")
    if errors:
        print("Errors:")
        for e in errors:
            print(f"  {e['record']}: {e['error']}")
    print(f"\nManifest written to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
