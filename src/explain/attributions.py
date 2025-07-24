import numpy as np
from tf_explain.core.integrated_gradients import IntegratedGradients
import shap

def compute_integrated_gradients(model, img_tensor, class_index, baseline=None):
    """
    model: tf.keras Model
    img_tensor: np.array shape (1, H, W, C), values [0–1]
    class_index: int, target class
    baseline: same shape as img_tensor or None to use zeros
    returns: np.array H×W saliency map (float32)
    """
    explainer = IntegratedGradients()
    data = (img_tensor, None)
    # tf‑explain wants a dict of {‘0’: model}, so we wrap:
    attributions = explainer.explain(
        data, 
        model, 
        class_index=class_index, 
        baseline=baseline, 
        n_steps=50
    )
    # attributions is H×W×C; collapse to grayscale
    ig_map = np.mean(attributions, axis=-1)
    # normalize to [0,1]
    ig_map -= ig_map.min()
    if ig_map.max() > 0:
        ig_map /= ig_map.max()
    return ig_map.astype("float32")

def compute_shap_values(model, background, img_tensor):
    """
    model: tf.keras Model
    background: np.array shape (B, H, W, C)
    img_tensor: np.array shape (1, H, W, C)
    returns: shap_values for each class: list of np.array (1, H, W, C)
    """
    # wrap model for SHAP
    f = lambda x: model(x).numpy()
    explainer = shap.DeepExplainer(f, background)
    shap_vals = explainer.shap_values(img_tensor)  
    # shap_vals is list of arrays per class
    return shap_vals
