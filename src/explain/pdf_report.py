import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PIL import Image
import numpy as np


def build_report_pdf(
    original_img: np.ndarray,
    saliency_img: np.ndarray,
    prediction: str,
    confidence: float,
    explanation: str,
) -> bytes:
    """
    Generates a PDF report containing the original image, saliency overlay,
    prediction, confidence, and explanation. Returns PDF as raw bytes.
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, height - 40, "Brain Tumor Classification Report")

    # Prediction and confidence
    c.setFont("Helvetica", 12)
    c.drawString(40, height - 70, f"Prediction: {prediction} ({confidence*100:.1f}%)")

    # Explanation text
    text_obj = c.beginText(40, height - 100)
    text_obj.setFont("Helvetica", 11)
    for line in explanation.splitlines():
        text_obj.textLine(line)
    c.drawText(text_obj)

    # Convert and resize images for PDF
    orig_pil = Image.fromarray(original_img)
    sal_pil  = Image.fromarray(saliency_img)
    max_w = width/2 - 60
    max_h = height/3
    orig_pil.thumbnail((max_w, max_h))
    sal_pil.thumbnail((max_w, max_h))

    # Draw images side by side
    y_pos = height/2 - orig_pil.height/2
    c.drawInlineImage(orig_pil, 40, y_pos)
    c.drawInlineImage(sal_pil, 60 + orig_pil.width, y_pos)

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.read()
