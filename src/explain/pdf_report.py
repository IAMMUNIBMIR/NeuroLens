# src/explain/pdf_report.py

import io
from typing import Optional, Sequence

import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader, simpleSplit
from reportlab.pdfgen import canvas


def build_report_pdf(original_img: np.ndarray, saliency_img: np.ndarray, label: str, confidence: float, explanation: str, probs: Optional[Sequence[float]] = None, labels: Optional[Sequence[str]] = None,) -> bytes:
    """
    Build a one‐page PDF report showing:
      - the original image
      - the saliency overlay
      - classification result + confidence
      - optional class‐probability table
      - text explanation
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    w, h = letter

    # Title
    c.setTitle("Brain Tumor Classification Report")
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, h - 50, "Brain Tumor Classification Report")

    # Helper: numpy array → in‐memory PNG
    def to_buffer(arr: np.ndarray) -> io.BytesIO:
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        buf.seek(0)
        return buf

    orig_buf = to_buffer(original_img)
    sal_buf  = to_buffer(saliency_img)

    # Compute sizes & positions
    margin = 40
    img_w  = (w - 3 * margin) / 2
    aspect = original_img.shape[0] / original_img.shape[1]
    img_h  = img_w * aspect
    img_y  = h - 80 - img_h

    # Draw images
    c.drawImage(ImageReader(orig_buf), margin, img_y, img_w, img_h)
    c.drawImage(ImageReader(sal_buf), 2*margin + img_w, img_y, img_w, img_h)

    # Prediction line
    text_y = img_y - 30
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, text_y, f"Prediction: {label} ({confidence*100:.2f}%)")

    # Optional probability table
    if probs is not None and labels is not None:
        c.setFont("Helvetica", 10)
        table_y = text_y - 20
        for i, (lab, p) in enumerate(sorted(zip(labels, probs), key=lambda x: -x[1])):
            c.drawString(margin + 20, table_y - 14*i, f"{lab:<12} {p:.4f}")
        text_y = table_y - 14 * len(labels) - 10
    else:
        text_y -= 30

    # Explanation block
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, text_y, "Explanation:")
    c.setFont("Helvetica", 10)
    max_width = w - 2*margin
    lines = simpleSplit(explanation, "Helvetica", 10, max_width)
    for i, line in enumerate(lines):
        c.drawString(margin, text_y - 14*(i+1), line)

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.read()
