# NeuroLens 🧠

Brain MRI tumor classification with **Integrated Gradients** & **SHAP** explanations.  
Deployed on **Streamlit Community Cloud**. **Not for clinical use.**

- **Live demo:** https://neurolens-munib.streamlit.app  
- **Model card:** [docs/model_card.md](docs/model_card.md)  
- **Dataset:** Public MRI image dataset (4 classes: *glioma, meningioma, no tumor, pituitary*)

---

## Why it matters
NeuroLens demonstrates a complete, **explainable** medical-imaging workflow end-to-end: data ingest (DICOM or PNG/JPG), model inference, attribution maps, lightweight telemetry, and a polished PDF report—**all in one click** and reproducible locally.

---

## Highlights
- 📥 **Inputs:** single **PNG/JPG** slice or full **DICOM** series (`.zip` / `.dcm`)
- 🧠 **Models:** **Custom CNN (224×224)** and **Xception (299×299)** (switchable at runtime)
- 🔎 **Explainability:** **Integrated Gradients** & **SHAP** overlays
- 📈 **Metrics & Diagnostics:** in-app **requests count**, **p50/p95 latency**, **softmax KL-drift** proxy, and **CSV export**
- 🧾 **PDF report:** prediction, class-probability table, and side-by-side images
- 🛡️ **Graceful UX:** friendly secret checks, rule-based text fallback when Gemini is unavailable

---

## Quickstart (local)

> Requires Python 3.11+.

```bash
pip install -r requirements.txt
streamlit run app.py
