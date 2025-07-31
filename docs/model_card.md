# NeuroLens – Model Card

**Task:** Brain MRI tumor classification (**Glioma / Meningioma / No tumor / Pituitary**)  
**Models:** Custom CNN (224×224), Xception (299×299, fine-tuned)  
**Intended use:** Educational/experimental demo to illustrate classification and explanations. **Not for clinical diagnosis.**

---

## 1. Overview
NeuroLens is a Streamlit app that accepts a brain MRI slice or a DICOM series, runs a tumor classifier, and provides post-hoc explanations (Integrated Gradients and SHAP). It is designed for demonstration and learning purposes.

---

## 2. Data
- **Source:** Public brain tumor MRI image dataset (four classes: *glioma, meningioma, pituitary, no tumor*) commonly distributed in a **Training** and **Testing** directory structure.
- **Class balance:** Four classes with moderate imbalance (typically “No tumor” slightly higher than others).
- **Preprocessing:**
  - **PNG/JPG slices:** resize to **224×224** (Custom CNN) or **299×299** (Xception), convert to RGB if needed, scale to **[0,1]**.
  - **DICOM series:** per-slice min–max normalization to uint8, resize to model input, replicate to 3 channels, scale to **[0,1]**.
- **Train/Val split:** Uses the dataset’s Train/Test split. Within **Train**, validation is created with **15%** via Keras `ImageDataGenerator(validation_split=0.15)`.

---

## 3. Model details
- **Custom CNN:** Convolutional blocks → Flatten → Dense(128) → **Dense(4, softmax)** at **224×224×3** input;  
  **Loss:** categorical cross-entropy; **Optimizer:** **Adamax (lr=0.001)**;  
  **Augmentation (when training):** rotation **15°**, width/height shift **0.04**, shear **0.05**, zoom **0.05**, horizontal flip; rescale **1/255**.
- **Xception:** Keras/TensorFlow Xception backbone with a fine-tuned head; **softmax over 4 classes**; input **299×299×3**;  
  **Loss:** categorical cross-entropy; **Optimizer:** Adamax (lr=0.001).

---

## 4. Intended users & use cases
- Students, researchers, and engineers exploring explainable AI for medical images.  
- Not intended for patient care or diagnostic decisions.

---

## 5. Limitations / Risks
- Dataset bias and variability (scanner/sequences/institutions) can affect generalization.  
- Post-hoc explanations can be noisy; highlighted pixels are not causal proof.  
- Single-slice inference may miss 3D context present in full volumes.

---

## 6. Interpretability
- **Integrated Gradients** and **SHAP** overlays are provided at reduced resolution for performance.  
- Explanations highlight pixels contributing to the predicted class.

---

## 7. Monitoring
- The app tracks **inference latency** (per request) and a **softmax KL-drift** proxy (prediction distribution vs. a baseline).  
- Drift baseline defaults to uniform (**1/4, 1/4, 1/4, 1/4** for four classes); can be overridden via `BASELINE_PROBS` (comma-separated) in secrets/env.

---

## 8. Ethical considerations
- The app displays a **“not for clinical use”** disclaimer.  
- Users should not upload PHI or identifiable data.

---

## 9. Versioning
- **App version:** `1.0.0` (also shown in the PDF header when enabled).  
- **Changelog:** See repository releases/tags for updates.
