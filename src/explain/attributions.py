# src/explain/attributions.py
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
    # sometimes expl is a dict, sometimes an ndarray
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
    # create zero‐baseline background of shape (1, H, W, C)
    background = np.zeros((1,) + img_tensor.shape[1:], dtype=img_tensor.dtype)

    # build a new Model that outputs only the logit/probability for our class_index
    single_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=tf.expand_dims(model(model.inputs)[:, class_index], axis=-1)
    )

    # now DeepExplainer will see a Keras Model with a single‐scalar output
    explainer = shap.DeepExplainer(single_model, background)
    # returns a list of length 1, each array shaped (1, H, W, C)
    shap_vals = explainer.shap_values(img_tensor, nsamples=nsamples)[0]

    # collapse colour channels and normalize
    heatmap = np.abs(shap_vals[0]).sum(-1)  # shape (H, W)
    return heatmap / (heatmap.max() + 1e-8)
