import io
import numpy as np
import pytest
from PIL import Image

try:
    mod = __import__("src.visualize.gif_csv", fromlist=["*"])
except Exception:
    mod = __import__("gif_csv", fromlist=["*"])

def _frame_count(gif_bytes: bytes) -> int:
    img = Image.open(io.BytesIO(gif_bytes))
    count = 0
    try:
        while True:
            img.seek(count)
            count += 1
    except EOFError:
        pass
    return count

def test_generate_slice_gif_matches_slice_count():
    S, H, W = 6, 32, 32
    rng = np.random.default_rng(0)
    volume = (rng.random((S, H, W)) * 255).astype("uint8")
    saliency = (rng.random((S, H, W))).astype("float32")

    gif_bytes = mod.generate_slice_gif(volume, saliency, duration=0.05)
    assert isinstance(gif_bytes, (bytes, bytearray)) and len(gif_bytes) > 0
    assert _frame_count(gif_bytes) == S

def test_build_slice_metrics_csv_schema():
    S, C = 5, 3
    rng = np.random.default_rng(1)
    logits = rng.random((S, C)).astype("float32")
    preds = logits / logits.sum(axis=1, keepdims=True)
    labels = ["Glioma", "Meningioma", "No tumor"]

    csv_text = mod.build_slice_metrics_csv(preds, labels)
    assert "slice_idx" in csv_text
    assert all(lab in csv_text for lab in labels)
    # 5 data rows (+ header)
    assert csv_text.strip().count("\n") >= S
