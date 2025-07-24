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
    Returns a normalized SHAP heatmap H×W for the given class_index.
    """
    # 1×H×W×C zero baseline
    background = np.zeros((1,) + img_tensor.shape[1:], dtype=img_tensor.dtype)

    # Build a Keras model that outputs only the class_index score
    class_score = tf.keras.layers.Lambda(
        lambda x: tf.expand_dims(x[:, class_index], axis=-1),
        output_shape=(1,),
        name="shap_class_select"
    )(model.output)
    single_model = tf.keras.Model(inputs=model.inputs, outputs=class_score)

    # Now DeepExplainer sees a single-output Keras model
    explainer = shap.DeepExplainer(single_model, background.astype(np.float32))
    # shap_vals: list of arrays (one per output), here just one element
    shap_vals = explainer.shap_values(img_tensor.astype(np.float32))
    arr = shap_vals[0] if isinstance(shap_vals, list) else shap_vals
    # arr has shape (1, H, W, C). Collapse batch and channels:
    heatmap = np.abs(arr[0]).sum(-1)
    return heatmap / (heatmap.max() + 1e-8)