import numpy as np
import tensorflow as tf
from keras.layers import ReLU

tf.keras.layers.ThresholdedReLU = ReLU

from tf_explain.core.integrated_gradients import IntegratedGradients
import shap

def compute_integrated_gradients(model, img_tensor, class_index, baseline=None):
    ig = IntegratedGradients()
    data = (img_tensor, None)
    expl = ig.explain(
        data,
        model,
        class_index,
        baseline=baseline or np.zeros_like(img_tensor),
        n_steps=50
    )
    heatmap = expl["attributions"]  # shape (H,W)
    heatmap = np.clip(heatmap, 0, None)
    return heatmap / (heatmap.max() + 1e-8)

def compute_shap_values(model, img_tensor, nsamples=50):
    # use deep explainer with a zero baseline
    background = np.zeros((1,) + img_tensor.shape[1:], dtype=img_tensor.dtype)
    explainer = shap.DeepExplainer((model, model.inputs), background)
    shap_vals = explainer.shap_values(img_tensor, nsamples=nsamples)
    # shap_vals is a list of arrays, one per class
    heatmaps = {}
    for cls_idx, arr in enumerate(shap_vals):
        # arr has shape (1,H,W,3) → collapse channels
        hm = np.abs(arr[0]).sum(-1)
        heatmaps[cls_idx] = hm / (hm.max() + 1e-8)
    return heatmaps
