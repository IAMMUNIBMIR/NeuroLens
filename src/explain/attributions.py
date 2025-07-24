# src/explain/attributions.py
import numpy as np
import shap
from tf_explain.core.integrated_gradients import IntegratedGradients

def compute_integrated_gradients(model, img_tensor, class_index, baseline=None):
    """
    Returns a normalized Integrated Gradients heatmap (H×W) for class_index.
    """
    ig = IntegratedGradients()
    data = (img_tensor, None)
    expl = ig.explain(
        data,
        model,
        class_index,
        n_steps=15,
    )

    # expl may be an array or a dict; normalize both
    if isinstance(expl, dict):
        heatmap = expl.get("attributions", next(iter(expl.values())))
    else:
        heatmap = expl

    heatmap = np.clip(heatmap, 0, None)
    return heatmap / (heatmap.max() + 1e-8)


def compute_shap_values(model, img_tensor, class_index, nsamples=50):
    """
    Returns a normalized SHAP heatmap (H×W) for class_index.
    """
    # 1×H×W×C zero baseline
    background = np.zeros((1,) + img_tensor.shape[1:], dtype=img_tensor.dtype)

    # wrap the model so DeepExplainer only sees a single‐output function
    def model_fn(x):
        preds = model(x)          # shape (batch, num_classes)
        # pull out just the class of interest
        # ensure a pure numpy array
        return preds[:, class_index]

    explainer = shap.DeepExplainer(model_fn, background)
    # shap_vals will be an array of shape (1, H, W, C)
    shap_vals = explainer.shap_values(img_tensor, nsamples=nsamples)

    # collapse channels and normalize
    arr = shap_vals
    heatmap = np.abs(arr[0]).sum(-1)   # H×W
    return heatmap / (heatmap.max() + 1e-8)
