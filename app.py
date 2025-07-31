import os, zipfile, tempfile, io
from pathlib import Path

import cv2
import numpy as np
import requests
import streamlit as st
import plotly.graph_objects as go
import PIL.Image
import tensorflow as tf
import hashlib
import requests

from keras.models import load_model, Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adamax
from keras.metrics import Precision, Recall
from keras.preprocessing import image
import google.generativeai as genai
from streamlit.runtime.scriptrunner import RerunException

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from dotenv import load_dotenv
from src.explain.fallback_text import compute_saliency_stats, rule_based_explanation
from src.explain.pdf_report import build_report_pdf
from src.visualize.gif_csv import generate_slice_gif, build_slice_metrics_csv
from src.visualize.slice_plots import plot_slice_bar_chart
import src.explain.attributions as attributions
from src.telemetry.metrics import get_store, timed_infer
tf.keras.backend.clear_session()

# ---------------------- NEW: model downloader + checksum + cached loader -----
from typing import Optional

def _sha256(path: Path) -> str:
    """Compute sha256 of a file and return 'sha256:<hex>'."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()

def _download_and_verify(url: str, dst: Path, expected_sha: Optional[str] = None) -> Path:
    """
    Download url -> dst atomically, verify sha256 if provided, and return dst.
    Reuses existing file if already present (and matches checksum when given).
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Reuse a good existing file
    if dst.exists() and dst.stat().st_size > 0:
        if expected_sha is None or _sha256(dst) == expected_sha:
            return dst
        else:
            dst.unlink(missing_ok=True)  # bad file; redownload

    # Download to a temp file first
    tmp = dst.with_suffix(dst.suffix + ".part")
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)

    # Verify checksum if provided
    if expected_sha is not None:
        actual = _sha256(tmp)
        if actual != expected_sha:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"Model checksum mismatch.\nExpected: {expected_sha}\nActual:   {actual}"
            )

    tmp.replace(dst)  # atomic move
    return dst

@st.cache_resource(show_spinner="Loading model…")
def get_model(kind: str = "cnn"):
    """
    Download + cache model file (once per container), verify sha256, load with Keras,
    and cache the loaded model object for reuse.
    """
    models_dir = Path("models"); models_dir.mkdir(parents=True, exist_ok=True)
    if kind == "xception":
        url  = st.secrets["MODEL_XCP_URL"]
        sha  = st.secrets.get("xception_model_sha")
        path = models_dir / "xception_full.keras"
    else:
        url  = st.secrets["MODEL_CNN_URL"]
        sha  = st.secrets.get("cnn_model_sha")
        path = models_dir / "cnn_model.keras"

    _download_and_verify(url, path, expected_sha=sha)
    return load_model(path, compile=False)
# -----------------------------------------------------------------------------

# ---------------------- constants/helpers --------------------------
LABELS = ['Glioma', 'Meningioma', 'No tumor', 'Pituitary']

SALIENCY_DIR = Path('saliency_maps'); SALIENCY_DIR.mkdir(exist_ok=True, parents=True)
MODELS_DIR   = Path('models');        MODELS_DIR.mkdir(exist_ok=True, parents=True)
MAX_FILE_BYTES = 250 * 1024 * 1024  # 250 MB

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
    with tf.GradientTape() as tape:
        img_tensor = tf.convert_to_tensor(img_array)
        tape.watch(img_tensor)
        # just call the model directly:
        predictions = model(img_tensor)
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
    if getattr(up, "size", 0) > MAX_FILE_BYTES:
        st.error(f"File too large ({up.size/1_048_576:.1f} MB). Max is 250 MB.")
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

with st.sidebar:
    st.header("About")
    st.markdown(
        "- **Live demo:** https://neurolens-munib.streamlit.app\n"
        "- **Model Card:** see `docs/model_card.md` in the repository\n"
        "- **Disclaimer:** Demo for education; **not for clinical use**."
    )

# --------------- CHANGED: use cached downloader/loader instead of local files
selected_model = st.radio("Select Model", ("Transfer Learning - Xception", "Custom CNN"))

if selected_model == "Transfer Learning - Xception":
    model = get_model("xception")
    img_size = (299, 299)
else:
    model = get_model("cnn")
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

            # NEW: progress + cancel
            S = vol_resized.shape[0]
            if "stop_all" not in st.session_state:
                st.session_state["stop_all"] = False
            cols = st.columns([1,1])
            stop_btn = cols[0].button("⏹ Stop")
            progress = cols[1].progress(0, text="Analyzing slices…")
            if stop_btn:
                st.session_state["stop_all"] = True

            preds = []
            saliency_stack = []
            for i in range(S):
                if st.session_state.get("stop_all"):
                    st.info(f"Stopped at slice {i}/{S}.")
                    break

                p = model.predict(vol_rgb[i:i+1], verbose=0)
                preds.append(p[0])

                sm = generate_saliency_map(
                    model,
                    vol_rgb[i:i+1],           # (1, H, W, 3)
                    np.argmax(p[0]),          # class index
                    img_size
                )
                saliency_stack.append(sm)

                progress.progress((i+1)/S, text=f"Analyzing slices… {i+1}/{S}")

            progress.empty()
            st.session_state["stop_all"] = False

            if len(preds) == 0:
                st.stop()

            preds = np.asarray(preds)                # (N, C)
            saliency_stack = np.asarray(saliency_stack)  # (N, H, W)

            # 2) generate GIF
            gif_bytes = generate_slice_gif(vol_resized[:len(saliency_stack)], saliency_stack, duration=0.1)
            st.image(
                gif_bytes,
                caption="3D walkthrough (auto‐loop)",
                output_format="GIF",
                use_column_width=True
            )

            # 3) generate CSV
            csv_str = build_slice_metrics_csv(preds, LABELS)
            st.download_button("Download slice metrics as CSV", data=csv_str, file_name="slice_metrics.csv", mime="text/csv")

            # 4) per-slice bar chart (for current slider index)
            fig2 = plot_slice_bar_chart(preds, slice_idx, LABELS)
            st.plotly_chart(fig2, use_container_width=True)

else:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None and getattr(uploaded_file, "size", 0) > MAX_FILE_BYTES:
        st.error(f"File too large ({uploaded_file.size/1_048_576:.1f} MB). Max is 250 MB.")
        st.stop()
    if not uploaded_file:
        st.stop()
    img = image.load_img(uploaded_file, target_size=img_size)
    img_array = image.img_to_array(img)
    original_img_for_display = ((img_array - img_array.min()) / (img_array.ptp()+1e-8) * 255).astype("uint8")
    img_array = np.expand_dims(img_array, axis=0).astype("float32") / 255.0

# ---------------------- Prediction & Visualization ---------------------------
store = get_store(st)
prediction, latency_ms = timed_infer(model.predict, img_array, verbose=0)
# record metrics (latency + drift proxy)
try:
    store.add(latency_ms, probs=prediction[0], labels=LABELS)
except Exception:
    pass

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
          <h3 style="margin-bottom:10px;font-size:20px;">Predicted probability</h3>
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

# Metrics & Diagnostics (no-cost, in-app)
with st.expander("Metrics & Diagnostics", expanded=False):
    summary = store.summary()
    if summary.get("count", 0) > 0:
        colA, colB, colC = st.columns(3)
        colA.metric("Requests", summary["count"])
        colB.metric("Median latency (ms)", f"{summary['p50_ms']:.1f}")
        colC.metric("P95 latency (ms)", f"{summary['p95_ms']:.1f}")

        # Latency timeline
        import plotly.graph_objects as go
        latencies = [r.latency_ms for r in store.rows]
        xs = list(range(1, len(latencies)+1))
        fig_lat = go.Figure()
        fig_lat.add_scatter(x=xs, y=latencies, mode="lines+markers", name="latency_ms")
        fig_lat.update_layout(height=240, margin=dict(l=10,r=10,t=10,b=10), yaxis_title="ms")
        st.plotly_chart(fig_lat, use_container_width=True)

        # KL-drift timeline (if available)
        drifts = [r.kl_drift for r in store.rows if r.kl_drift is not None]
        if drifts:
            xs2 = list(range(1, len(drifts)+1))
            fig_kl = go.Figure()
            fig_kl.add_scatter(x=xs2, y=drifts, mode="lines+markers", name="softmax_kl_drift")
            fig_kl.update_layout(height=240, margin=dict(l=10,r=10,t=10,b=10), yaxis_title="KL")
            st.plotly_chart(fig_kl, use_container_width=True)

        st.download_button("Download metrics CSV", data=store.to_csv(), file_name="metrics.csv", mime="text/csv")
    else:
        st.caption("Run a prediction to populate metrics.")

slice_for_text = slice_idx if mode.startswith("DICOM") else None
explanation = safe_explanation(saliency_map_path,result,float(prediction[0][class_index]),saliency_map,probabilities,LABELS,slice_for_text)
st.write("## Explanation")
st.write(explanation)

pdf_bytes = build_report_pdf(original_img_for_display, superimposed_img, result, float(prediction[0][class_index]), explanation, probs=prediction[0], labels=LABELS)
st.download_button("📄 Download Report as PDF", data=pdf_bytes, file_name="brain_tumor_report.pdf", mime="application/pdf")

tabs = st.tabs(["Integrated Gradients", "SHAP"])

with tabs[0]:
    ig_map = attributions.compute_integrated_gradients(model, img_array, class_index)
    heat_ig = cv2.applyColorMap((ig_map*255).astype("uint8"), cv2.COLORMAP_VIRIDIS)
    overlay_ig = (0.6*heat_ig + 0.4*original_img_for_display).astype("uint8")
    st.image(overlay_ig, caption="Integrated Gradients", use_container_width=True)

with tabs[1]:
    x_for_shap = img_array.astype("float32", copy=False)
    try:
        shap_map = attributions.compute_shap_values(model, x_for_shap, class_index)
        heat_shap = cv2.applyColorMap((shap_map*255).astype("uint8"), cv2.COLORMAP_PLASMA)
        overlay_shap = (0.6*heat_shap + 0.4*original_img_for_display).astype("uint8")
        st.image(overlay_shap, caption="SHAP DeepExplainer", use_container_width=True)
    except Exception as e:
        st.warning(f"SHAP failed on this input: {e}")
        st.caption("Showing Integrated Gradients overlay instead.")
        st.image(overlay_ig, caption="Integrated Gradients (fallback)", use_container_width=True)
