import types
import tempfile
from pathlib import Path
import numpy as np
import pytest
import importlib
import io

# Import module whether it's in src.data or repo root
try:
    loader = importlib.import_module("src.data.dicom_loader")
except Exception:
    loader = importlib.import_module("dicom_loader")


def _write_fake_dcm(path: Path):
    """
    Write a tiny file that passes loader._is_dicom:
    the last 4 bytes at offset 128..131 must be b"DICM".
    """
    header = bytearray(132)
    header[128:132] = b"DICM"
    path.write_bytes(bytes(header))


def test_basic_dicom_flow_with_mock(monkeypatch):
    # Monkeypatch pydicom in the loader module to avoid the real dependency
    class FakeDS:
        def __init__(self, instance_no=1):
            self.Rows = 16
            self.Columns = 16
            self.SamplesPerPixel = 1
            self.BitsAllocated = 16
            self.InstanceNumber = instance_no
            self.SeriesInstanceUID = "SERIES-123"
            self.PixelData = (np.ones((16, 16), dtype="uint16")
                              * (50 + instance_no)).tobytes()

        @property
        def pixel_array(self):
            return np.frombuffer(self.PixelData, dtype="uint16").reshape(16, 16)

    def fake_dcmread(path, *args, **kwargs):
        # Derive instance number from filename suffix if present
        try:
            stem = Path(path).stem
            digits = "".join(ch for ch in stem if ch.isdigit())
            inst = int(digits) if digits else 1
        except Exception:
            inst = 1
        return FakeDS(instance_no=inst)

    fake_pydicom = types.SimpleNamespace(dcmread=fake_dcmread)
    monkeypatch.setattr(loader, "pydicom", fake_pydicom, raising=True)

    # Build a temporary series folder that passes _is_dicom
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for i in (1, 2, 3):
            _write_fake_dcm(d / f"IM_{i:03d}.dcm")

        # Call the internal series loader
        fn = getattr(loader, "_load_series", None)
        if fn is None:
            pytest.skip("No _load_series found on dicom_loader")
        vol = fn(d)

    assert isinstance(vol, np.ndarray)
    assert vol.ndim == 3  # (S, H, W)
    assert vol.shape[0] >= 3
    assert float(vol.max()) > 0  # not all zeros


def test_load_zip_via_public_api(monkeypatch):
    """Exercise the public zip API: load_dicom_zip(bytes) -> (S,H,W) volume."""
    class FakeDS:
        def __init__(self, instance_no=1):
            self.Rows = 16
            self.Columns = 16
            self.SamplesPerPixel = 1
            self.BitsAllocated = 16
            self.InstanceNumber = instance_no
            self.SeriesInstanceUID = "SERIES-123"
            self.PixelData = (np.ones((16, 16), dtype="uint16")
                              * (60 + instance_no)).tobytes()

        @property
        def pixel_array(self):
            return np.frombuffer(self.PixelData, dtype="uint16").reshape(16, 16)

    def fake_dcmread(path, *args, **kwargs):
        try:
            stem = Path(path).stem
            digits = "".join(ch for ch in stem if ch.isdigit())
            inst = int(digits) if digits else 1
        except Exception:
            inst = 1
        return FakeDS(instance_no=inst)

    fake_pydicom = types.SimpleNamespace(dcmread=fake_dcmread)
    monkeypatch.setattr(loader, "pydicom", fake_pydicom, raising=True)

    # Create a zip with 3 fake DICOM files that pass _is_dicom
    import zipfile
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for i in (1, 2, 3):
            data = bytearray(132)
            data[128:132] = b"DICM"
            z.writestr(f"series/IM_{i:03d}.dcm", bytes(data))
    zip_bytes = mem.getvalue()

    # Use your public API name here (not load_zipped_dicom)
    fn = getattr(loader, "load_dicom_zip", None)
    if fn is None:
        pytest.skip("load_dicom_zip not available on dicom_loader")
    vol = fn(zip_bytes)

    assert isinstance(vol, np.ndarray)
    assert vol.ndim == 3 and vol.shape[0] >= 3
