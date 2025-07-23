import numpy as np

def compute_saliency_stats(smap: np.ndarray, high_q=90):
    """Return dict of basic stats from a 2D saliency map (0-1-ish)."""
    smap = smap.astype(float)
    thr = np.percentile(smap, high_q)
    mask = smap >= thr
    frac_high = float(mask.mean())  # % of pixels above high threshold
    if mask.sum() == 0:
        return {"frac_high": 0.0, "quadrant": "none", "centroid": (0, 0), "thr": thr}

    ys, xs = np.where(mask)
    h, w = smap.shape
    cx, cy = xs.mean() / w, ys.mean() / h  # normalized centroid (0-1)
    # crude quadrant labels for axial view
    lr = "left" if cx < 0.5 else "right"
    ap = "anterior" if cy < 0.5 else "posterior"
    return {
        "frac_high": frac_high,
        "quadrant": f"{ap}-{lr}",
        "centroid": (float(cx), float(cy)),
        "thr": float(thr)
    }

def rule_based_explanation(pred_label: str,
                           confidence: float,
                           stats: dict,
                           top2: list[tuple[str, float]],
                           slice_idx: int | None):
    """Return <=4 sentences of human-readable text, no ML jargon."""
    frac_pct = stats["frac_high"] * 100
    loc = stats["quadrant"].replace("-", " ").title() if stats["quadrant"] != "none" else "no focal area"
    s_idx = f" on slice {slice_idx}" if slice_idx is not None else ""
    alt = ""
    if len(top2) > 1 and top2[0][0] != top2[1][0]:
        alt = f" The next most likely class is {top2[1][0]} ({top2[1][1]*100:.1f}%)."

    lines = [
        f"The model predicts **{pred_label}** with {confidence*100:.1f}% confidence{s_idx}.",
        f"About {frac_pct:.1f}% of the pixels exceeded the saliency threshold, mainly in the **{loc}** region.",
        "These highlighted areas are where signal intensity patterns best matched the training examples for this class.",
        f"Interpretation should be combined with radiologist review and full clinical context.{alt}"
    ]
    return " ".join(lines[:4])
