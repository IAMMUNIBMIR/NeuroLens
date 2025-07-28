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
import src.explain.attributions as attributions
from src.seg import segment
tf.keras.backend.clear_session()

APP_VERSION = "1.0.0"

# ---------------------- constants/helpers --------------------------
LABELS = ['Glioma', 'Meningioma', 'Pituitary']
IMG_SIZE_CNN = (224, 224)
IMG_SIZE_XCP = (299, 299)
MAX_UPLOAD_SIZE_MB = 200

def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _download_and_verify(url, dest, expected_sha=None):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and expected_sha:
        if _sha256(dest) == expected_sha.split(":")[-1]:
            return dest
    if not url:
        raise ValueError("Model URL not provided.")
    import requests
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for c in r.iter_content(8192):
            f.write(c)
    if expected_sha and _sha256(dest) != expected_sha.split(":")[-1]:
        raise ValueError("SHA mismatch for downloaded model.")
    return dest

@st.cache_resource(show_spinner="Loading model…")
def get_model(kind: str = "cnn"):
    models_dir = Path("models"); models_dir.mkdir(parents=True, exist_ok=True)
    try:
        if kind == "xception":
            url  = st.secrets["MODEL_XCP_URL"]
            sha  = st.secrets.get("xception_model_sha")
            path = models_dir / "xception_full.keras"
        else:
            url  = st.secrets["MODEL_CNN_URL"]
            sha  = st.secrets.get("cnn_model_sha")
            path = models_dir / "cnn_model.keras"
    except KeyError as e:
        missing = str(e).strip("'")
        st.error(
            f"Missing secret `{missing}`. Add it in **.streamlit/secrets.toml**:\n\n"
            "```toml\n"
            "MODEL_CNN_URL = \"https://.../cnn_model.keras\"\n"
            "MODEL_XCP_URL = \"https://.../xception_full.keras\"\n"
            "cnn_model_sha = \"sha256:...\"        # optional\n"
            "xception_model_sha = \"sha256:...\"   # optional\n"
            "```\n"
        )
        st.stop()

    _download_and_verify(url, path, expected_sha=sha)
    return load_model(path, compile=False)

load_dotenv()

# ---- Gemini robust config (avoid metadata 503) ----
ENABLE_GEMINI = True
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") if hasattr(st, "secrets") else None
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    ENABLE_GEMINI = False

def safe_explanation(img_path, pred_label, confidence, saliency_map, probs, labels, slice_idx=None):
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
        st.caption("Using rule‑based explanation because the AI explainer was unavailable.")
        return rule_based_explanation(pred_label, confidence, stats, top2, slice_idx)

# ---------------------- UI --------------------------
st.set_page_config(page_title="NeuroLens", layout="wide")
st.title("🧠 NeuroLens — Brain MRI Classification with Explanations")

uploaded = st.file_uploader("Upload a DICOM ZIP or a single PNG/JPG slice", type=["zip", "png", "jpg", "jpeg"])

selected_model = st.selectbox("Select model", ["Custom CNN", "Transfer Learning - Xception"])
if selected_model.startswith("Transfer"):
    model = get_model("xception")
    img_size = IMG_SIZE_XCP
else:
    model = get_model("cnn")
    img_size = IMG_SIZE_CNN

def prep_slice(slice_gray, img_size):
    rgb = np.repeat(cv2.resize(slice_gray, img_size)[..., None], 3, axis=-1)
    rgb_disp = (rgb * 255).astype("uint8") if rgb.max() <= 1.0 else rgb.astype("uint8")
    model_in = np.expand_dims(rgb_disp / 255.0, 0).astype("float32")
    return model_in, rgb_disp

if uploaded is not None:
    name = uploaded.name.lower()
    if name.endswith(".zip"):
        from src.data.dicom_loader import load_dicom_zip
        volume = load_dicom_zip(uploaded.read())
        slice_idx = st.slider("Slice index", 0, int(volume.shape[0]-1), int(volume.shape[0]//2))
        img_array, original_img_for_display = prep_slice(volume[slice_idx], img_size)

        run_all = st.checkbox("Analyze all slices", value=False)
        MAX_SLICES_WARN = 200
        if run_all and volume.shape[0] > MAX_SLICES_WARN:
            st.warning(f"This series has {volume.shape[0]} slices. Running the model on all slices may be slow.")
            proceed_all = st.checkbox(f"I understand—run on all {volume.shape[0]} slices", key="confirm_run_all")
            if not proceed_all:
                run_all = False

        if run_all:
            with st.spinner("Running model on all slices..."):
                vol_resized = np.stack([cv2.resize(s, img_size) for s in volume], axis=0)
                vol_rgb     = np.repeat(vol_resized[..., None], 3, axis=-1) / 255.0
                preds       = model.predict(vol_rgb, verbose=0)

                # 1) saliency for each slice
                saliency_stack = []
                for i in range(vol_resized.shape[0]):
                    ig_map = attributions.compute_integrated_gradients(model, np.expand_dims(np.repeat(vol_resized[i][..., None],3,axis=-1)/255.0,0), int(np.argmax(preds[i])))
                    saliency_stack.append(ig_map)
                saliency_stack = np.asarray(saliency_stack)

                # 2) export GIF + CSV
                gif_bytes = generate_slice_gif(vol_resized, saliency_stack, duration=0.05)
                csv_text  = build_slice_metrics_csv(preds, LABELS)

                st.download_button("🎞️ Download slices GIF", data=gif_bytes, file_name="slices.gif")
                st.download_button("📊 Download slice metrics CSV", data=csv_text, file_name="slice_metrics.csv")
        else:
            pass
    else:
        img = PIL.Image.open(uploaded).convert("L")
        arr = np.array(img)
        volume = np.expand_dims(arr, 0)  # single-slice volume
        slice_idx = 0
        img_array, original_img_for_display = prep_slice(volume[slice_idx], img_size)

    prediction = model.predict(img_array, verbose=0)
    class_index = int(np.argmax(prediction[0]))
    result = LABELS[class_index]
    explanation = "Explanation not available."

    # Placeholder: save saliency map image to a temp PNG for Gemini prompt
    ig_map = attributions.compute_integrated_gradients(model, img_array, class_index)
    heat_ig = cv2.applyColorMap((ig_map*255).astype("uint8"), cv2.COLORMAP_OCEAN)
    superimposed_img = cv2.addWeighted(original_img_for_display, 0.7, heat_ig, 0.3, 0)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        PIL.Image.fromarray(superimposed_img).save(tmp.name)
        tmp_path = tmp.name

    try:
        explanation = safe_explanation(tmp_path, result, float(prediction[0][class_index]), ig_map, prediction[0], LABELS, slice_idx)
    except Exception as e:
        st.warning(f"Explanation generation failed: {e}")

    # Headline cards
    st.markdown(f"""
    <div style="display:flex;gap:12px;align-items:center;background:#0b1d2a;padding:16px;border-radius:12px;color:#fff;">
      <div style="flex:1;text-align:center;">
        <h3 style="margin-bottom:10px;font-size:20px;">Prediction</h3>
        <p style="font-size:36px;font-weight:800;color:#4CAF50;margin:0;">{result}</p>
      </div>
      <div style="width:2px;height:80px;background-color:#ffffff;margin:0 20px;"></div>
      <div style="flex:1;text-align:center;">
        <h3 style="margin-bottom:10px;font-size:20px;">Predicted probability</h3>
        <p style="font-size:36px;font-weight:800;color:#2196F3;margin:0;">{prediction[0][class_index]:.4%}</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Prob bar chart
    fig = go.Figure()
    fig.add_bar(x=LABELS, y=prediction[0].tolist())
    fig.update_layout(height=280, margin=dict(l=10,r=10,t=10,b=10), yaxis=dict(tickformat=".0%"))
    st.plotly_chart(fig, use_container_width=True)

    # Explanation text
    st.write(explanation)

    # PDF report
    model_name = "Xception" if selected_model.startswith("Transfer") else "Custom CNN"
    pdf_bytes = build_report_pdf(
        original_img_for_display, superimposed_img, label=result,
        confidence=float(prediction[0][class_index]),
        explanation=explanation, probs=prediction[0], labels=LABELS,
        model_name=model_name, app_version=APP_VERSION
    )
    st.download_button("📄 Download Report as PDF", data=pdf_bytes, file_name="brain_tumor_report.pdf", mime="application/pdf")

    tabs = st.tabs(["Integrated Gradients", "SHAP"])

    with tabs[0]:
        st.image(superimposed_img, caption="Integrated Gradients overlay", use_column_width=True)

    with tabs[1]:
        shap_map = attributions.compute_shap_values(model, img_array, class_index)
        heat_shap = cv2.applyColorMap((shap_map*255).astype("uint8"), cv2.COLORMAP_OCEAN)
        super_shap = cv2.addWeighted(original_img_for_display, 0.7, heat_shap, 0.3, 0)
        st.image(super_shap, caption="SHAP overlay", use_column_width=True)
