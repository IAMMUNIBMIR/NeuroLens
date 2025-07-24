import numpy as np
import tensorflow as tf
import shap
from tf_explain.core.integrated_gradients import IntegratedGradients

if not hasattr(tf.keras.backend, "learning_phase"):
    tf.keras.backend.learning_phase = lambda: 0

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
    Returns a normalized H×W SHAP heatmap for the given class_index
    using shap.GradientExplainer.
    """
    # 1×H×W×C zero baseline
    background = np.zeros((1,) + img_tensor.shape[1:], dtype=img_tensor.dtype)

    # wrap your model so SHAP only sees the single-class score
    def model_fn(x):
        preds = model(x)              # shape (batch, num_classes)
        return preds[:, class_index]  # shape (batch,)

    explainer = shap.GradientExplainer(model_fn, background)
    # returns list-of-length‑1, each array shape (batch, H, W, C)
    shap_vals = explainer.shap_values(img_tensor, batch_size=1)

    # collapse channels → H×W, take absolute and normalize
    arr = shap_vals[0]             # (1, H, W, C)
    heatmap = np.abs(arr[0]).sum(-1)
    return heatmap / (heatmap.max() + 1e-8)