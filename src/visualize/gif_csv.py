import io
import numpy as np
import pandas as pd
import imageio
import cv2

def generate_slice_gif(
    volume: np.ndarray,           # shape (S, H, W), uint8 or float in [0,1]
    saliency_maps: np.ndarray,    # shape (S, H, W), float in [0,1]
    duration: float = 0.1         # seconds per frame
) -> bytes:
    """
    Build an in‑memory GIF stepping through each slice, overlaying saliency.
    Returns raw GIF bytes.
    """
    frames = []
    for i in range(volume.shape[0]):
        # normalize & stack original slice to RGB
        sl = volume[i]
        if sl.dtype != np.uint8:
            sl = ((sl - sl.min()) / (sl.ptp() + 1e-8) * 255).astype(np.uint8)
        rgb = np.stack([sl, sl, sl], axis=-1)

        # build heatmap from saliency
        heat = (saliency_maps[i] * 255).astype(np.uint8)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

        # blend heatmap + original
        frame = (0.7 * heat + 0.3 * rgb).astype(np.uint8)
        frames.append(frame)

    buf = io.BytesIO()
    imageio.mimsave(buf, frames, format='GIF', duration=duration)
    buf.seek(0)
    return buf.getvalue()


def build_slice_metrics_csv(
    preds: np.ndarray,            # shape (S, C)
    labels: list[str],            # e.g. ['Glioma','Meningioma',…]
) -> str:
    """
    Build a CSV string with one row per slice:
      slice_idx, pred_class, class_probabilities…
    """
    n_slices, n_classes = preds.shape
    top_idx = np.argmax(preds, axis=1)
    rows = []
    for i in range(n_slices):
        row = {
            "slice_idx": i,
            "pred_class": labels[top_idx[i]],
        }
        # include all class probabilities
        for c, lab in enumerate(labels):
            row[lab] = preds[i, c]
        rows.append(row)

    df = pd.DataFrame(rows)
    return df.to_csv(index=False)
