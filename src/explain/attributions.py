import numpy as np
import tensorflow as tf
import shap
from tf_explain.core.integrated_gradients import IntegratedGradients

def compute_integrated_gradients(model, img_tensor, class_index, baseline=None):
    ig = IntegratedGradients()
    expl = ig.explain(
        (img_tensor, None),
        model,
        class_index,
        n_steps=15,
    )
    if isinstance(expl, dict):
        heatmap = expl.get("attributions", next(iter(expl.values())))
    else:
        heatmap = expl
    heatmap = np.clip(heatmap, 0, None)
    return heatmap / (heatmap.max() + 1e-8)


def compute_shap_values(model, img_tensor, class_index, nsamples=50):
    """
    Returns a normalized SHAP heatmap (H×W) for class_index.
    Uses GradientExplainer under TF‑2 (no need to wrap in a Keras Model).
    """
    # 1×H×W×C zero baseline
    background = np.zeros((1,) + img_tensor.shape[1:], dtype=img_tensor.dtype)

    # use the gradient explainer (eager mode)
    explainer = shap.GradientExplainer(model, background)
    # this returns a list of arrays, one per class, each shape (1,H,W,C)
    all_shap_vals = explainer.shap_values(img_tensor, nsamples=nsamples)
    # pick out the one for our class:
    shap_for_class = all_shap_vals[class_index]  # shape (1, H, W, C)

    # collapse color channels and normalize to [0,1]
    heatmap = np.abs(shap_for_class[0]).sum(-1)   # now H×W
    return heatmap / (heatmap.max() + 1e-8)

