"""Inspect the NIFECGDB .edf/.qrs files to determine channel layout,
sampling rate, and annotation structure before designing the pipeline.
"""
from pathlib import Path

import mne
import wfdb

mne.set_log_level("ERROR")

DATA_DIR = Path(__file__).resolve().parent.parent / "non-invasive-fetal-ecg-database-1.0.0" / "non-invasive-fetal-ecg-database-1.0.0"


def list_records() -> list[str]:
    records_file = DATA_DIR / "RECORDS"
    names = []
    for line in records_file.read_text().splitlines():
        line = line.strip()
        if line:
            names.append(line.removesuffix(".edf"))
    return names


def inspect_edf(record: str) -> dict:
    edf_path = DATA_DIR / f"{record}.edf"
    raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose="ERROR")
    info = {
        "record": record,
        "n_channels": len(raw.ch_names),
        "labels": raw.ch_names,
        "sample_rates": [raw.info["sfreq"]] * len(raw.ch_names),
        "duration_s": raw.n_times / raw.info["sfreq"],
        "n_samples": [raw.n_times] * len(raw.ch_names),
    }
    return info


def inspect_qrs(record: str) -> dict:
    ann = wfdb.rdann(str(DATA_DIR / record), extension="edf.qrs")
    return {
        "record": record,
        "n_annotations": len(ann.sample),
        "first_samples": ann.sample[:5].tolist(),
        "symbols": sorted(set(ann.symbol)),
    }


def main() -> None:
    records = list_records()
    print(f"Found {len(records)} records\n")

    # Sample a handful of records to check whether layout is consistent
    sample = records[:5] + records[-2:]
    label_sets = set()
    rate_sets = set()

    for record in sample:
        info = inspect_edf(record)
        label_sets.add(tuple(info["labels"]))
        rate_sets.add(tuple(info["sample_rates"]))
        print(f"{record}: channels={info['n_channels']} labels={info['labels']} "
              f"fs={info['sample_rates']} duration={info['duration_s']:.1f}s")

    print(f"\nDistinct label layouts across sample: {len(label_sets)}")
    for labels in label_sets:
        print(f"  {labels}")
    print(f"Distinct sample-rate layouts across sample: {len(rate_sets)}")
    for rates in rate_sets:
        print(f"  {rates}")

    print("\n--- QRS annotation check ---")
    for record in sample[:3]:
        try:
            qinfo = inspect_qrs(record)
            print(f"{record}: n_annotations={qinfo['n_annotations']} "
                  f"symbols={qinfo['symbols']} first_samples={qinfo['first_samples']}")
        except Exception as e:
            print(f"{record}: QRS read failed ({e})")


if __name__ == "__main__":
    main()
