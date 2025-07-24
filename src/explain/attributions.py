# src/explain/attributions.py
import numpy as np
import tensorflow as tf
from tf_explain.core.integrated_gradients import IntegratedGradients

# ---------------- Integrated Gradients (as you already had) ---------------- #
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


# ---------------- “GradSHAP‑lite”: SHAP‑style but no SHAP dependency -------- #
def compute_shap_values(
    model,
    img_tensor,
    class_index,
    nsamples: int = 20,
    baseline: str = "zeros",
):
    """
    A lightweight SHAP-like attribution (gradient * (x - baseline)) averaged
    over multiple random baselines (if baseline == 'random'). This avoids the
    incompatibilities between shap and TF/Keras 3, but gives you a similar
    qualitative visualization.

    Returns: (H, W) normalized heatmap in [0, 1].
    """
    x = tf.convert_to_tensor(img_tensor, dtype=tf.float32)  # (1,H,W,C)

    def make_baseline():
        if baseline == "zeros":
            return tf.zeros_like(x)
        elif baseline == "mean":
            return tf.fill(tf.shape(x), tf.reduce_mean(x))
        elif baseline == "random":
            return tf.random.uniform(tf.shape(x), 0.0, 1.0, dtype=tf.float32)
        else:
            # default to zeros if unknown
            return tf.zeros_like(x)

    heat_sum = None

    for _ in range(nsamples):
        b = make_baseline()
        with tf.GradientTape() as tape:
            tape.watch(x)
            preds = model(x, training=False)[:, class_index]  # (1,)
        grads = tape.gradient(preds, x)  # (1,H,W,C)

        # SHAP-ish attribution: gradient * (x - baseline)
        contrib = grads * (x - b)
        # collapse channels, abs and reduce to HxW
        heat = tf.reduce_sum(tf.math.abs(contrib), axis=-1)[0]  # (H,W)

        if heat_sum is None:
            heat_sum = heat.numpy()
        else:
            heat_sum += heat.numpy()

    heatmap = heat_sum / nsamples
    heatmap = heatmap / (heatmap.max() + 1e-8)
    return heatmap
