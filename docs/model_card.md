# NeuroLens – Model Card

**Task:** Brain MRI tumor classification (Glioma / Meningioma / Pituitary)  
**Models:** Custom CNN (224×224), Xception (299×299, fine‑tuned)  
**Intended use:** Educational/experimental demo to illustrate classification and explanations. **Not for clinical diagnosis.**

---

## 1. Overview
NeuroLens is a Streamlit app that accepts a brain MRI slice or a DICOM series, runs a tumor classifier, and provides post‑hoc explanations (Integrated Gradients and SHAP). It is designed for demonstration and learning purposes.

## 2. Data
- **Source:** [document the dataset you used; e.g., Kaggle brain MRI dataset or internal sample].
- **Preprocessing:** grayscale → resize to model input (CNN: 224×224, Xception: 299×299); scale to [0,1]; for series, works per‑slice.
- **Class balance:** [insert approximate class counts or note if imbalanced].
- **Train/Val split:** [insert split method].

## 3. Metrics (validation)
Report on your held‑out validation set:
- **Accuracy:** [..]
- **Per‑class F1:** Glioma [..], Meningioma [..], Pituitary [..]
- **ROC‑AUC (macro):** [..]
- **Latency (CPU, Streamlit Cloud):** median [..] ms, P95 [..] ms (from in‑app metrics).

> Note: Metrics reported here are not indicative of clinical performance.

## 4. Model details
- **Custom CNN:** [brief architecture summary if available].
- **Xception:** Keras/TensorFlow Xception base with top layers fine‑tuned; softmax over 3 classes.
- **Loss:** categorical cross‑entropy; **Optimizer:** [e.g., Adamax]; **Augmentation:** [if any].

## 5. Intended users & use cases
- Students, researchers, and engineers exploring explainable AI for medical images.
- Not intended for patient care or diagnostic decisions.

## 6. Limitations / Risks
- Dataset may be small or biased; may not generalize across scanners, sequences, or institutions.
- Post‑hoc explanations can be noisy; saliency does not imply causal importance.
- Single‑slice inference may miss 3D context present in full volumes.

## 7. Interpretability
- **Integrated Gradients** and **SHAP** overlays are provided at reduced resolution for performance.
- Explanations highlight pixels contributing to the predicted class.

## 8. Monitoring
- The app tracks **inference latency** and a simple **softmax KL‑drift** proxy to detect distributional shifts.
- Drift baseline defaults to uniform (1/3,1/3,1/3); can be overridden via `BASELINE_PROBS` secret/env var.

## 9. Ethical considerations
- The app displays a **“not for clinical use”** disclaimer.
- Users should not upload PHI or identifiable data.

## 10. Versioning
- **App version:** `1.0.0` (shown in the PDF header)
- **Changelog:** See repository releases/tags for updates.
