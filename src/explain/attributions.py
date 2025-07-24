import numpy as np
import keras
from keras.layers import ReLU

keras.layers.ThresholdedReLU = ReLU

from tf_explain.core.integrated_gradients import IntegratedGradients
import shap

def compute_integrated_gradients(model, img_tensor, class_index, baseline=None):
    """
    model: tf.keras.Model
    img_tensor: numpy array shape (1,H,W,3)
    class_index: int
    baseline: same shape as img_tensor or None
    returns: numpy array (H,W) scaled [0..1]
    """
    ig = IntegratedGradients()
    data = (img_tensor, None)  # second arg is labels placeholder
    expl = ig.explain(data,
                      model,
                      class_index,
                      baseline=baseline or np.zeros_like(img_tensor),
                      n_steps=50)
    heatmap = expl["attributions"]  # shape (H,W)
    # normalize
    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max() + 1e-8
    return heatmap

def compute_shap_values(model, img_tensor, nsamples=50):
    """
    model: tf.keras.Model
    img_tensor: numpy array shape (1,H,W,3)
    nsamples: int
    returns: dict[class_idx → numpy array (H,W)]
    """
    # use a tiny set of background samples (here zeros)
    background = np.zeros((1,) + img_tensor.shape[1:], dtype=img_tensor.dtype)
    explainer = shap.DeepExplainer((model, model.inputs), background)
    shap_values = explainer.shap_values(img_tensor, nsamples=nsamples)
    # shap_values is a list of C arrays each (1,H,W,3)
    heatmaps = {}
    for cls_idx, arr in enumerate(shap_values):
        # sum across channels & normalize
        hm = np.abs(arr[0]).sum(-1)
        hm /= hm.max() + 1e-8
        heatmaps[cls_idx] = hm
    return heatmaps
