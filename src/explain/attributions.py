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
    if isinstance(expl, dict):
        heatmap = expl.get("attributions", next(iter(expl.values())))
    else:
        heatmap = expl
    heatmap = np.clip(heatmap, 0, None)
    return heatmap / (heatmap.max() + 1e-8)


def compute_shap_values(model, img_tensor, class_index, nsamples=50):
    """
    Returns a normalized H×W SHAP heatmap for class_index.
    """
    # zero‐baseline of same H×W×C
    background = np.zeros((1,) + img_tensor.shape[1:], dtype=img_tensor.dtype)

    # build a Keras Model that outputs only the selected class’s score
    # slicing inside a Lambda needs an explicit output_shape=(1,)
    class_output = tf.keras.layers.Lambda(
        lambda x: tf.expand_dims(x[:, class_index], axis=-1),
        output_shape=(1,),
        name="shap_class_select"
    )(model.output)
    single_model = tf.keras.Model(inputs=model.inputs, outputs=class_output)

    explainer = shap.DeepExplainer(single_model, background)
    # shap_vals[0] has shape (1, H, W, C)
    shap_vals = explainer.shap_values(img_tensor, nsamples=nsamples)[0]

    # collapse channels and normalize
    heatmap = np.abs(shap_vals[0]).sum(-1)
    return heatmap / (heatmap.max() + 1e-8)
