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
    Returns a normalized H×W SHAP heatmap for the given class_index,
    using shap.GradientExplainer on a single-output Keras Model.
    """
    # 1×H×W×C zero baseline
    background = np.zeros((1,) + img_tensor.shape[1:], dtype=img_tensor.dtype)

    # 1) carve out just the score for our class from the multi‑class model:
    class_output = Lambda(
        lambda x: tf.expand_dims(x[:, class_index], axis=-1),
        output_shape=(1,),
        name="shap_class_select"
    )(model.output)

    single_model = Model(inputs=model.inputs, outputs=class_output)

    # 2) build the explainer on that single-output model
    explainer = shap.GradientExplainer(single_model, background)

    # 3) get attributions (no batch_size arg)
    shap_vals = explainer.shap_values(img_tensor)

    # 4) collapse channels → (H, W), take absolute & normalize
    arr     = shap_vals[0][0]          # from list-of-length‑1
    heatmap = np.abs(arr).sum(-1)      # shape (H, W)
    return heatmap / (heatmap.max() + 1e-8)