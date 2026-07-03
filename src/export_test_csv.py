"""Export a slice of a real held-out test record to CSV, in the exact format
the web app expects (3 abdominal channels, no header, 1000Hz), for
end-to-end testing of the browser inference pipeline outside the browser."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_pipeline import ROOT, load_record

OUT_PATH = ROOT / "web" / "test_sample.csv"


def main() -> None:
    splits = json.loads((ROOT / "data" / "splits.json").read_text())
    record = splits["test"][0]
    rec = load_record(record)

    duration_s = 20
    n = int(duration_s * rec.fs)
    data = rec.abdomen[:, :n]  # (3, n)

    with OUT_PATH.open("w") as f:
        for i in range(n):
            f.write(",".join(f"{data[c, i]:.8e}" for c in range(3)) + "\n")

    n_true_qrs = int(((rec.qrs_samples >= 0) & (rec.qrs_samples < n)).sum())
    print(f"record={record}, wrote {n} samples ({duration_s}s) to {OUT_PATH}")
    print(f"true fetal QRS count in this slice: {n_true_qrs}")


if __name__ == "__main__":
    main()
