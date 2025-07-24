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
    Returns a normalized SHAP heatmap (H×W) for the given class_index.
    Uses the new shap.Explainer interface.
    """
    # 1×H×W×C zero baseline
    background = np.zeros((1,) + img_tensor.shape[1:], dtype=img_tensor.dtype)

    # 1) build the Explainer once
    explainer = shap.Explainer(model, background)

    # 2) explain your single example (max_evals trades off speed vs. accuracy)
    explanation = explainer(img_tensor, max_evals=nsamples)

    # 3) get the raw SHAP values array
    #    - explanation.values has shape (1, H, W, C)
    vals = explanation.values  # np.ndarray

    # 4) grab just your class_index channel and collapse any channels
    #    (for e.g. RGB input you'd do sum over the last axis, etc.)
    #    Here `vals[0,…,class_index]` is H×W
    heatmap = np.abs(vals[0, ..., class_index])

    # 5) normalize
    return heatmap / (heatmap.max() + 1e-8)