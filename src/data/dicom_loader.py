import zipfile, tempfile, os
from pathlib import Path
import numpy as np
import pydicom

def _is_dicom(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(132)[128:132] == b"DICM"
    except Exception:
        return False

def _load_series(folder: Path):
    """Return the largest coherent series (by InstanceNumber) in a folder."""
    files = [p for p in folder.rglob("*") if p.is_file() and _is_dicom(p)]
    if not files:
        raise ValueError("No DICOM files found in archive")
    # group by SeriesInstanceUID
    series_map = {}
    for f in files:
        d = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
        uid = getattr(d, "SeriesInstanceUID", "unknown")
        series_map.setdefault(uid, []).append(f)
    # pick largest series
    uid, slices = max(series_map.items(), key=lambda kv: len(kv[1]))
    # sort by InstanceNumber or SliceLocation fallback
    def sort_key(p):
        d = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
        return getattr(d, "InstanceNumber", getattr(d, "SliceLocation", 0))
    slices = sorted(slices, key=sort_key)
    # read pixels
    arrays = []
    first = pydicom.dcmread(str(slices[0]))
    intercept = float(getattr(first, "RescaleIntercept", 0))
    slope     = float(getattr(first, "RescaleSlope", 1))
    for s in slices:
        ds = pydicom.dcmread(str(s))
        arr = ds.pixel_array.astype(np.float32)
        arr = arr * slope + intercept
        arrays.append(arr)
    volume = np.stack(arrays, axis=0)  # (S, H, W)
    return volume

def load_dicom_zip(file_bytes: bytes) -> np.ndarray:
    """Load DICOM series from a zip uploaded via Streamlit FileUploader."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        zpath = tmpdir / "upload.zip"
        with open(zpath, "wb") as f:
            f.write(file_bytes)
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(tmpdir)
        vol = _load_series(tmpdir)
    return vol

def load_single_dcm(file_bytes: bytes) -> np.ndarray:
    ds = pydicom.dcmread(file_bytes)
    arr = ds.pixel_array.astype(np.float32)
    return arr[None, ...]  # shape (1, H, W)
