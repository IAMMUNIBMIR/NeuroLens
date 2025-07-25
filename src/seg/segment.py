import numpy as np
import nibabel as nib

def _try_imports():
    import torch  # noqa: F401
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor  # noqa: F401

def run_nnunet(volume: np.ndarray, spacing_mm=None, model_dir: str | None = None) -> np.ndarray:
    """
    volume: (S, H, W) float32 or uint8
    returns: binary mask (S, H, W) uint8 {0,1}
    """
    _try_imports()
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    # nnU-Net expects (C, Z, Y, X) float32
    vol = volume.astype(np.float32)[None, ...]  # add channel dim

    # configure predictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        verbose=False
    )
    # You can either point to a model folder you ship, or to an installed plan id.
    # For MVP assume you ship a folder in models/nnunet_brats/
    model_dir = model_dir or "models/nnunet_brats"
    predictor.initialize_from_trained_model_folder(
        model_dir,
        use_folds=(0,),  # one fold is enough for MVP
        checkpoint_name="checkpoint_final.pth"
    )

    with torch.no_grad():
        logits = predictor.predict_sliding_window_return_logits(vol)[0]  # (classes, Z, Y, X)

    # pick tumour class (assume class 1 == tumour for your checkpoint)
    mask = (np.argmax(logits, axis=0) > 0).astype(np.uint8)
    return mask  # (S,H,W)

def overlay(volume_u8: np.ndarray, mask: np.ndarray, alpha=0.4) -> np.ndarray:
    """Return RGB overlay for a single slice (H,W,3)."""
    import cv2
    h, w = mask.shape
    base = volume_u8 if volume_u8.ndim == 3 else np.stack([volume_u8]*3, -1)
    color = np.zeros_like(base)
    color[..., 0] = 255  # red
    mask3 = np.stack([mask]*3, -1)
    return (base*(1-alpha) + color*alpha*mask3).astype(np.uint8)

def estimate_volume_cc(mask: np.ndarray, spacing_mm=None) -> float | None:
    """Return tumour volume in cc if spacing is known, else None."""
    voxels = int(mask.sum())
    if spacing_mm is None:
        return None
    vx = float(spacing_mm[0] * spacing_mm[1] * spacing_mm[2])  # mm^3
    return voxels * vx / 1000.0  # cc

def export_nifti(mask: np.ndarray, affine=None) -> bytes:
    import io
    buf = io.BytesIO()
    img = nib.Nifti1Image(mask.astype(np.uint8), affine if affine is not None else np.eye(4))
    nib.save(img, buf)
    return buf.getvalue()
