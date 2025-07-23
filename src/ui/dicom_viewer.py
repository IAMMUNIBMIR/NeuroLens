import streamlit as st
import numpy as np
from typing import Literal
from src.data.dicom_loader import load_dicom_zip, load_single_dcm

def normalize_slice(slice2d: np.ndarray) -> np.ndarray:
    # simple min-max; you can add windowing later
    s = slice2d
    s = (s - np.min(s)) / (np.max(s) - np.min(s) + 1e-8)
    return (s * 255).astype("uint8")

def get_oriented(volume: np.ndarray, orientation: Literal["axial","sagittal","coronal"]) -> np.ndarray:
    if orientation == "axial":    # (S, H, W)
        return volume
    if orientation == "sagittal": # (W, S, H)
        return np.swapaxes(volume, 0, 2)
    if orientation == "coronal":  # (H, S, W)
        return np.swapaxes(volume, 0, 1)
    raise ValueError("bad orientation")

@st.cache_data(show_spinner=False)
def read_dicom(file_bytes: bytes, ext: str) -> np.ndarray:
    if ext == ".zip":
        return load_dicom_zip(file_bytes)
    return load_single_dcm(file_bytes)

def dicom_uploader_and_viewer():
    up = st.file_uploader("Upload MRI DICOM (.zip or .dcm)", type=["zip","dcm"])
    if not up:
        return None, None, None

    vol = read_dicom(up.getvalue(), f".{up.name.split('.')[-1].lower()}")
    st.success(f"Loaded volume shape: {vol.shape}")  # (S,H,W)

    orientation = st.radio("Orientation", ("axial","sagittal","coronal"), horizontal=True)
    oriented = get_oriented(vol, orientation)
    idx = st.slider("Slice", 0, oriented.shape[0]-1, oriented.shape[0]//2, 1)
    slice_img = normalize_slice(oriented[idx])

    st.image(slice_img, caption=f"{orientation} slice {idx}/{oriented.shape[0]-1}", use_column_width=True)
    return vol, orientation, idx
