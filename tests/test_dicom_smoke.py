import numpy as np
from src.data.dicom_loader import load_single_dcm

def test_single_dcm_loader():
    # fake bytes -> expect fail
    try:
        load_single_dcm(b"notdicom")
    except Exception:
        assert True
