import os, zipfile, tempfile, io
from pathlib import Path

import cv2
import numpy as np
import requests
import streamlit as st
import plotly.graph_objects as go
import PIL.Image
import tensorflow as tf

from keras.models import load_model, Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adamax
from keras.metrics import Precision, Recall
from keras.preprocessing import image
import google.generativeai as genai

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from dotenv import load_dotenv
from src.explain.fallback_text import compute_saliency_stats, rule_based_explanation
from src.explain.pdf_report import build_report_pdf
from src.visualize.gif_csv import generate_slice_gif, build_slice_metrics_csv
from src.visualize.slice_plots import plot_slice_bar_chart
tf.keras.backend.clear_session()

# ---------------------- constants/helpers --------------------------
LABELS = ['Glioma', 'Meningioma', 'No tumor', 'Pituitary']

SALIENCY_DIR = Path('saliency_maps'); SALIENCY_DIR.mkdir(exist_ok=True, parents=True)
MODELS_DIR   = Path('models');        MODELS_DIR.mkdir(exist_ok=True, parents=True)

def fetch_if_missing(url_env_key: str, dest: Path):
    if dest.exists():
        return dest
    url = os.getenv(url_env_key)
    if not url:
        raise ValueError(f"{url_env_key} not set in environment/secrets")
    r = requests.get(url, stream=True); r.raise_for_status()
    with open(dest, "wb") as f:
        for c in r.iter_content(8192):
            f.write(c)
    return dest

load_dotenv()

# ---- Gemini robust config (avoid metadata 503) ----
ENABLE_GEMINI = True

# pull key from secrets or env
GOOGLE_API_KEY = None
if hasattr(st, "secrets"):
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY:
    os.environ["NO_GCE_CHECK"] = "1"                 # stop metadata probe
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    import google.generativeai as genai
    genai.configure(api_key=GOOGLE_API_KEY, transport="rest")


def safe_explanation(img_path, pred_label, confidence, saliency_map,probs, labels, slice_idx=None):

    """Try Gemini; on failure or disabled, fall back to rule-based explanation."""
    stats = compute_saliency_stats(saliency_map)
    top2_idx = np.argsort(probs)[::-1][:2]
    top2 = [(labels[i], float(probs[i])) for i in top2_idx]

    if not ENABLE_GEMINI:
        return rule_based_explanation(pred_label, confidence, stats, top2, slice_idx)

    prompt = f"""You are an expert neurologist...
    The machine learning model predicted the image to of class "{pred_label}" with a confidence of {confidence*100}%.
    In your response:
    - Explain what regions of the brain the model is focusing on (light cyan).
    - Explain possible reasons for the prediction.
    - Do not restate how saliency maps work.
    - Max 4 sentences."""

    try:
        img = PIL.Image.open(img_path).resize((512, 512))
        model = genai.GenerativeModel("gemini-1.5-flash")
        resp = model.generate_content([prompt, img], request_options={"timeout": 40})
        return resp.text
    except Exception as e:
        st.warning(f"Gemini failed: {e}")
        return rule_based_explanation(pred_label, confidence, stats, top2, slice_idx)


def generate_saliency_map(model, img_array, class_index, img_size):
    img_tensor = tf.convert_to_tensor(img_array)
    with tf.name_scope("saliency"):
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            # force inference mode & isolate in its own name_scope
            with tf.name_scope("model_forward"):
                predictions = model(img_tensor, training=False)
                target_class = predictions[:, class_index]

    gradients = tape.gradient(target_class, img_tensor)
    gradients = tf.math.abs(gradients)
    gradients = tf.reduce_max(gradients, axis=-1).numpy().squeeze()
    gradients = cv2.resize(gradients, img_size)

    center = (gradients.shape[0] // 2, gradients.shape[1] // 2)
    radius = min(center) - 10
    y, x = np.ogrid[:gradients.shape[0], :gradients.shape[1]]
    mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2

    brain_grad = gradients[mask]
    if brain_grad.max() > brain_grad.min():
        brain_grad = (brain_grad - brain_grad.min()) / (brain_grad.max() - brain_grad.min())
    gradients[mask] = brain_grad

    thr = np.percentile(gradients[mask], 80)
    gradients[gradients < thr] = 0
    gradients = cv2.GaussianBlur(gradients, (11, 11), 0)
    return gradients

def load_model_custom(model_path):
    img_shape=(299,299,3)
    base_model = tf.keras.applications.Xception(include_top=False, weights="imagenet",
                                                input_shape=img_shape, pooling='max')
    model = Sequential([
        base_model, Flatten(),
        Dropout(0.3), Dense(128, activation='relu'),
        Dropout(0.25), Dense(4, activation='softmax')
    ])
    model.build((None,) + img_shape)
    model.compile(Adamax(0.001), loss='categorical_crossentropy',
                  metrics=['accuracy', Precision(), Recall()])
    model.load_weights(model_path)
    return model

# ---------------------- DICOM helpers -----------------------------------
def _is_dicom(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(132)[128:132] == b"DICM"
    except Exception:
        return False

def _load_series(folder: Path) -> np.ndarray:
    import pydicom
    files = [p for p in folder.rglob("*") if p.is_file() and _is_dicom(p)]
    if not files:
        raise ValueError("No DICOM files found")

    series_map = {}
    for f in files:
        d = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
        uid = getattr(d, "SeriesInstanceUID", "unknown")
        series_map.setdefault(uid, []).append(f)
    _, slices = max(series_map.items(), key=lambda kv: len(kv[1]))

    def sort_key(p):
        d = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
        return getattr(d, "InstanceNumber", getattr(d, "SliceLocation", 0))
    slices = sorted(slices, key=sort_key)

    first = pydicom.dcmread(str(slices[0]))
    intercept = float(getattr(first, "RescaleIntercept", 0))
    slope     = float(getattr(first, "RescaleSlope", 1))

    arrays = []
    for s in slices:
        ds = pydicom.dcmread(str(s))
        arr = ds.pixel_array.astype(np.float32) * slope + intercept
        arrays.append(arr)
    return np.stack(arrays, axis=0)  # (S,H,W)

@st.cache_data(show_spinner=False)
def load_dicom_zip(file_bytes: bytes) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        zpath = tmpdir / "upload.zip"
        with open(zpath, "wb") as f:
            f.write(file_bytes)
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(tmpdir)
        vol = _load_series(tmpdir)
    return vol

@st.cache_data(show_spinner=False)
def load_single_dcm(file_bytes: bytes) -> np.ndarray:
    import pydicom
    ds = pydicom.dcmread(io.BytesIO(file_bytes))
    arr = ds.pixel_array.astype(np.float32)
    return arr[None, ...]

def normalize_slice(slice2d: np.ndarray) -> np.ndarray:
    s = (slice2d - slice2d.min()) / (slice2d.ptp() + 1e-8)
    return (s * 255).astype("uint8")

def dicom_uploader_and_viewer():
    up = st.file_uploader("Upload MRI DICOM (.zip or .dcm)", type=["zip","dcm"])
    if not up:
        return None, None, None
    ext = up.name.lower().split(".")[-1]
    vol = load_dicom_zip(up.getvalue()) if ext == "zip" else load_single_dcm(up.getvalue())

    st.success(f"Loaded volume shape: {vol.shape}")  # (S,H,W)
    orientation = "axial"
    default_idx = st.session_state.pop("slice_slider_force", vol.shape[0]//2)
    idx = st.slider("Slice", 0, vol.shape[0]-1, default_idx, 1, key="slice_slider")
    slice_img = normalize_slice(vol[idx])
    st.image(slice_img, caption=f"Axial slice {idx}/{vol.shape[0]-1}", use_container_width=True)
    return vol, orientation, idx

# ---------------------- UI ---------------------------------------------------
st.title("Brain Tumor Classification")
st.write("Upload an image of a brain MRI scan to classify.")

@st.cache_resource
def get_models():
    # clear any lingering state
    tf.keras.backend.clear_session()
    xcep = tf.keras.models.load_model("xception_full.keras", compile=False)
    cnn  = tf.keras.models.load_model("cnn_model.keras",   compile=False)
    return xcep, cnn

selected_model = st.radio("Select Model", ("Transfer Learning - Xception", "Custom CNN"))

xcep_model, cnn_model = get_models()
if selected_model == "Transfer Learning - Xception":
    model = xcep_model
    img_size = (299, 299)
else:
    model = cnn_model
    img_size = (224, 224)

mode = st.radio("Input type", ["Image (PNG/JPG)", "DICOM (.zip/.dcm)"], horizontal=True)

if mode == "DICOM (.zip/.dcm)":
    volume, orientation, slice_idx = dicom_uploader_and_viewer()
    if volume is None:
        st.stop()

    def prep_slice(slice2d, size):
        s = cv2.resize(slice2d, size)
        disp = ((s - s.min()) / (s.ptp() + 1e-8) * 255).astype("uint8")
        rgb_disp = np.stack([disp, disp, disp], axis=-1)
        model_in = np.expand_dims(rgb_disp / 255.0, 0).astype("float32")
        return model_in, rgb_disp

    img_array, original_img_for_display = prep_slice(volume[slice_idx], img_size)

    run_all = st.checkbox("Analyze all slices", value=False)
    if run_all:
        with st.spinner("Running model on all slices..."):
            vol_resized = np.stack([cv2.resize(s, img_size) for s in volume], axis=0)
            vol_rgb     = np.repeat(vol_resized[..., None], 3, axis=-1) / 255.0
            preds       = model.predict(vol_rgb, verbose=0)

            # 1) build the saliency stack for every slice
            saliency_stack = []
            for i in range(vol_resized.shape[0]):
                sm = generate_saliency_map(
                    model,
                    vol_rgb[i:i+1],           # (1, H, W, 3)
                    np.argmax(preds[i]),      # class index
                    img_size
                )
                saliency_stack.append(sm)
            saliency_stack = np.stack(saliency_stack, axis=0)  # (S, H, W)

            # 2) generate GIF
            gif_bytes = generate_slice_gif(vol_resized, saliency_stack, duration=0.1)
            st.image(
                gif_bytes,
                caption="3D walkthrough (auto‐loop)",
                output_format="GIF",
                use_container_width=True
            )

            # 3) generate CSV
            csv_str = build_slice_metrics_csv(preds, LABELS)
            st.download_button("Download slice metrics as CSV", data=csv_str, file_name="slice_metrics.csv", mime="text/csv")

            # 4) per‑slice bar chart (for current slider index)
            fig2 = plot_slice_bar_chart(preds, slice_idx, LABELS)
            st.plotly_chart(fig2, use_container_width=True)

        # KEEP “most suspicious slice” logic, without the line chart above
        no_tumor_idx  = LABELS.index("No tumor")
        tumor_probs   = 1.0 - preds[:, no_tumor_idx]
        suspicious_idx = int(tumor_probs.argmax())
        st.write(f"Most suspicious slice: {suspicious_idx}")
        if st.button("Jump to that slice"):
            st.session_state["slice_slider_force"] = suspicious_idx
            st.experimental.rerun()

else:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if not uploaded_file:
        st.stop()
    img = image.load_img(uploaded_file, target_size=img_size)
    img_array = image.img_to_array(img)
    original_img_for_display = ((img_array - img_array.min()) / (img_array.ptp()+1e-8) * 255).astype("uint8")
    img_array = np.expand_dims(img_array, axis=0).astype("float32") / 255.0

# ---------------------- Prediction & Visualization ---------------------------
prediction = model.predict(img_array, verbose=0)

class_index = int(np.argmax(prediction[0]))
result = LABELS[class_index]

saliency_map = generate_saliency_map(model, img_array, class_index, img_size)

heatmap = cv2.applyColorMap(np.uint8(255 * saliency_map), cv2.COLORMAP_JET)
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
heatmap = cv2.resize(heatmap, img_size)

superimposed_img = (heatmap * 0.7 + original_img_for_display * 0.3).astype(np.uint8)

if mode != "DICOM (.zip/.dcm)":
    img_path = SALIENCY_DIR / uploaded_file.name
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
else:
    img_path = SALIENCY_DIR / f"dicom_slice_{slice_idx}.png"
    cv2.imwrite(str(img_path), cv2.cvtColor(original_img_for_display, cv2.COLOR_RGB2BGR))

saliency_map_path = str(SALIENCY_DIR / f"sal_{img_path.name}")
cv2.imwrite(saliency_map_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))

col1, col2 = st.columns(2)
with col1:
    st.image(original_img_for_display, caption='Input Image', use_container_width=True)
with col2:
    st.image(superimposed_img, caption='Saliency Map', use_container_width=True)

st.write("## Classification Results")
result_container = st.container()
result_container.markdown(
    f"""
    <div style="background-color:#000;color:#fff;padding:30px;border-radius:15px;">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <div style="flex:1;text-align:center;">
          <h3 style="margin-bottom:10px;font-size:20px;">Prediction</h3>
          <p style="font-size:36px;font-weight:800;color:#FF0000;margin:0;">{result}</p>
        </div>
        <div style="width:2px;height:80px;background-color:#ffffff;margin:0 20px;"></div>
        <div style="flex:1;text-align:center;">
          <h3 style="margin-bottom:10px;font-size:20px;">Confidence</h3>
          <p style="font-size:36px;font-weight:800;color:#2196F3;margin:0;">{prediction[0][class_index]:.4%}</p>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

probabilities = prediction[0]
sorted_idx = np.argsort(probabilities)[::-1]
sorted_labels = [LABELS[i] for i in sorted_idx]
sorted_probs = probabilities[sorted_idx]

fig = go.Figure(go.Bar(
    x=sorted_probs,
    y=sorted_labels,
    orientation='h',
    marker_color=['red' if lbl == result else 'blue' for lbl in sorted_labels]
))
fig.update_layout(
    title='Probabilities for each class',
    xaxis_title='Probability',
    yaxis_title='Class',
    height=400,
    width=600,
    yaxis=dict(autorange="reversed")
)
for i, prob in enumerate(sorted_probs):
    fig.add_annotation(x=prob, y=i, text=f'{prob:.4f}', showarrow=False, xanchor='left', xshift=5)

st.plotly_chart(fig)

slice_for_text = slice_idx if mode.startswith("DICOM") else None
explanation = safe_explanation(saliency_map_path,result,float(prediction[0][class_index]),saliency_map,probabilities,LABELS,slice_for_text)
st.write("## Explanation")
st.write(explanation)

pdf_bytes = build_report_pdf(original_img_for_display, superimposed_img, result, float(prediction[0][class_index]), explanation, probs=prediction[0], labels=LABELS)
st.download_button("📄 Download Report as PDF", data=pdf_bytes, file_name="brain_tumor_report.pdf", mime="application/pdf")