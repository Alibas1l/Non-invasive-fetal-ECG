"""Export the trained FetalECGUNet checkpoint to ONNX (float32), then
INT8-quantize it for a small, fast, browser-deployable model.

Two heads are exported ('wave', 'qrs') so the browser can show both the
denoised waveform and the fetal-QRS detection probability. The time axis is
a dynamic dimension so the same model handles any CSV upload length without
re-exporting, as long as it's a multiple of 16 (the U-Net has 4 pooling
stages, each halving the length -- see model.py's depth=4).
"""
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from model import FetalECGUNet

ROOT = Path(__file__).resolve().parent.parent
CKPT_PATH = ROOT / "data" / "fetal_ecg_unet.pt"
ONNX_FP32_PATH = ROOT / "web" / "public" / "model_fp32.onnx"
ONNX_INT8_PATH = ROOT / "web" / "public" / "model_int8.onnx"

IN_CHANNELS = 3
EXPORT_WINDOW_LEN = 2000  # must match training window_len; must be a multiple of 2**depth


def export_fp32() -> None:
    model = FetalECGUNet(in_channels=IN_CHANNELS, base_channels=16, depth=4)
    model.load_state_dict(torch.load(CKPT_PATH, map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, IN_CHANNELS, EXPORT_WINDOW_LEN)
    ONNX_FP32_PATH.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (dummy,),
        str(ONNX_FP32_PATH),
        input_names=["input"],
        output_names=["wave", "qrs"],
        dynamic_axes={
            "input": {0: "batch", 2: "time"},
            "wave": {0: "batch", 1: "time"},
            "qrs": {0: "batch", 1: "time"},
        },
        opset_version=17,
        dynamo=False,
    )
    onnx.checker.check_model(str(ONNX_FP32_PATH))
    print(f"Exported fp32 ONNX to {ONNX_FP32_PATH} ({ONNX_FP32_PATH.stat().st_size / 1024:.1f} KB)")


def quantize_int8() -> None:
    quantize_dynamic(
        model_input=str(ONNX_FP32_PATH),
        model_output=str(ONNX_INT8_PATH),
        weight_type=QuantType.QInt8,
    )
    print(f"Quantized INT8 ONNX to {ONNX_INT8_PATH} ({ONNX_INT8_PATH.stat().st_size / 1024:.1f} KB)")


def verify_parity() -> None:
    """Compare fp32 vs int8 ONNX outputs against each other on random input,
    to catch export/quantization bugs before this ever reaches the browser."""
    x = np.random.randn(1, IN_CHANNELS, EXPORT_WINDOW_LEN).astype(np.float32)

    sess_fp32 = ort.InferenceSession(str(ONNX_FP32_PATH))
    sess_int8 = ort.InferenceSession(str(ONNX_INT8_PATH))

    out_fp32 = sess_fp32.run(None, {"input": x})
    out_int8 = sess_int8.run(None, {"input": x})

    for name, o_fp32, o_int8 in zip(["wave", "qrs"], out_fp32, out_int8):
        mae = np.mean(np.abs(o_fp32 - o_int8))
        corr = np.corrcoef(o_fp32.flatten(), o_int8.flatten())[0, 1]
        print(f"{name}: MAE={mae:.4f}, correlation={corr:.4f}")


if __name__ == "__main__":
    export_fp32()
    quantize_int8()
    verify_parity()

    fp32_kb = ONNX_FP32_PATH.stat().st_size / 1024
    int8_kb = ONNX_INT8_PATH.stat().st_size / 1024
    print(f"\nSize reduction: {fp32_kb:.1f} KB -> {int8_kb:.1f} KB ({100 * (1 - int8_kb / fp32_kb):.1f}% smaller)")
