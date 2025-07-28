import io
import os
import numpy as np
import pytest

# Keep Streamlit/GUI libs quiet if app.py ever gets imported accidentally
os.environ.setdefault("STREAMLIT_HEADLESS", "1")

@pytest.fixture
def tiny_gray_stack():
    # 6 small 32x32 grayscale frames (uint8)
    rng = np.random.default_rng(0)
    frames = [(rng.random((32, 32)) * 255).astype("uint8") for _ in range(6)]
    return frames

@pytest.fixture
def tiny_rgb_image():
    rng = np.random.default_rng(1)
    return (rng.random((64, 64, 3)) * 255).astype("uint8")

@pytest.fixture
def tiny_probs():
    # 3-class softmax-ish vector
    p = np.array([0.2, 0.5, 0.3], dtype="float32")
    return p / p.sum()

@pytest.fixture
def tf_and_keras():
    try:
        import tensorflow as tf
        from keras import layers, models  # Keras 3 API
    except Exception:
        pytest.skip("TensorFlow/Keras not available", allow_module_level=True)
    return tf, layers, models

@pytest.fixture
def tiny_keras_model(tf_and_keras):
    tf, layers, models = tf_and_keras
    # Very small model; accepts 64x64x3, outputs 3-class probs
    inp = layers.Input(shape=(64, 64, 3))
    x = layers.Rescaling(1/255.0)(inp)
    x = layers.Conv2D(4, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Dense(3, activation="softmax")(x)
    model = models.Model(inp, out)
    model.compile()  # no training needed for smoke tests
    return model
