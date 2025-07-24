import numpy as np
import tensorflow as tf
import shap
from tf_explain.core.integrated_gradients import IntegratedGradients
from keras.layers     import Lambda
from keras.models     import Model

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
    using shap.GradientExplainer on a single‐output Keras model.
    """
    # zero baseline batch of shape (1,H,W,C)
    background = np.zeros((1,) + img_tensor.shape[1:], dtype=img_tensor.dtype)

    # 1) slice out just the class_index score from your multi‐class model:
    class_output = Lambda(
        lambda x: tf.expand_dims(x[:, class_index], axis=-1),
        output_shape=(1,),
        name="shap_class_select"
    )(model.output)

    single_model = Model(inputs=model.inputs, outputs=class_output)

    # 2) build the GradientExplainer on that single‐output model
    explainer = shap.GradientExplainer(single_model, background)

    # 3) get attributions: list of length 1, each array (batch, H, W, C)
    shap_vals = explainer.shap_values(img_tensor, batch_size=1)

    # 4) collapse to H×W, take absolute and normalize
    arr     = shap_vals[0][0]           # (H, W, C)
    heatmap = np.abs(arr).sum(-1)       # (H, W)
    return heatmap / (heatmap.max() + 1e-8)