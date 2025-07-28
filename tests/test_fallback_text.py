import numpy as np
import pytest

try:
    mod = __import__("src.explain.fallback_text", fromlist=["*"])
except Exception:
    mod = __import__("fallback_text", fromlist=["*"])

def test_rule_based_explanation_contains_label():
    smap = np.zeros((32, 32), dtype="float32")
    smap[8:16, 10:20] = 1.0  # a bright blob -> nonzero stats
    stats = mod.compute_saliency_stats(smap)
    text = mod.rule_based_explanation(
        pred_label="Meningioma",
        confidence=0.76,
        stats=stats,
        top2=[("Meningioma", 0.76), ("Glioma", 0.18)],
        slice_idx=12,
    )
    assert isinstance(text, str) and "Meningioma" in text and "%" in text
