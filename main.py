"""
FastAPI OCR Server for Salesforce Integration
Handles: image validation, OpenCV preprocessing, OCR (TrOCR for handwriting,
Tesseract fallback for printed text), field extraction, structured JSON response.
"""

import base64
import io
import re
from typing import Optional

import cv2
import numpy as np
import pytesseract
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

app = FastAPI(title="Salesforce OCR Server", version="1.0")

# ---------------------------------------------------------------------------
# Model loading (once, at startup)
# ---------------------------------------------------------------------------
print("Loading TrOCR model... (first run downloads ~1.3GB)")
TROCR_PROCESSOR = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
TROCR_MODEL = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
print("TrOCR model loaded.")

MAX_FILE_SIZE_MB = 8
ALLOWED_FORMATS = {"JPEG", "JPG", "PNG"}


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class OCRRequest(BaseModel):
    image_base64: str          # base64 encoded image (no data: prefix)
    file_name: Optional[str] = "document"
    # Default "printed": full-page Tesseract. This form is printed labels +
    # handwritten fill-ins — TrOCR expects single cropped text lines, not a
    # whole page, so it is NOT a good full-page engine for this layout.
    mode: Optional[str] = "printed"  # "handwritten" | "printed"


class ExtractedFields(BaseModel):
    installed_location_name: Optional[str] = None   # Installed_Location_Name__c
    installation_code: Optional[str] = None          # Installation_Code__c e.g. "00158-C252"
    installation_date: Optional[str] = None          # Installation_Date__c, normalized YYYY-MM-DD
    installation_location: Optional[str] = None       # Installation_Location__c (address)
    model_name: Optional[str] = None                  # used by Apex for Order_Product__c lookup


class OCRResponse(BaseModel):
    success: bool
    raw_text: str = ""
    fields: ExtractedFields = ExtractedFields()
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Preprocessing (OpenCV)
# ---------------------------------------------------------------------------
def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """Deskew, denoise, binarize for cleaner OCR input."""
    img = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # Adaptive threshold (binarize)
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 11
    )

    # Deskew based on text bounding box
    coords = np.column_stack(np.where(thresh < 255))
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) > 0.5:  # only rotate if meaningfully skewed
            (h, w) = thresh.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            thresh = cv2.warpAffine(
                thresh, M, (w, h),
                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
            )

    return thresh


# ---------------------------------------------------------------------------
# OCR engines
# ---------------------------------------------------------------------------
def run_trocr(processed_img: np.ndarray) -> str:
    """Run handwriting OCR via TrOCR. Expects single-line/region crops for
    best accuracy; here run on whole doc as a simple baseline."""
    pil_img = Image.fromarray(processed_img).convert("RGB")
    pixel_values = TROCR_PROCESSOR(images=pil_img, return_tensors="pt").pixel_values
    generated_ids = TROCR_MODEL.generate(pixel_values, max_length=256)
    text = TROCR_PROCESSOR.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text


def run_tesseract(processed_img: np.ndarray) -> str:
    """Run printed-text OCR via Tesseract."""
    return pytesseract.image_to_string(processed_img)


# ---------------------------------------------------------------------------
# Field extraction — tuned for ALAN Electronic Systems
# "Installation Report-cum-Warranty Certificate" layout ONLY.
# ---------------------------------------------------------------------------
def _normalize_date_ddmmyy(raw: str) -> Optional[str]:
    """Convert DD/MM/YY or DD/MM/YYYY -> YYYY-MM-DD. Indian format assumed."""
    raw = raw.strip().replace(" ", "")
    m = re.match(r"(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})", raw)
    if not m:
        return None
    day, month, year = m.groups()
    if len(year) == 2:
        year = "20" + year
    try:
        day_i, month_i = int(day), int(month)
        if not (1 <= day_i <= 31 and 1 <= month_i <= 12):
            return None
        return f"{year}-{month_i:02d}-{day_i:02d}"
    except ValueError:
        return None


def extract_fields(text: str) -> ExtractedFields:
    fields = ExtractedFields()

    # --- Hospital/Clinic Name -> Installed_Location_Name__c ---
    m = re.search(r"Hospital\s*/?\s*Clinic\s*Name\s*[:\-]\s*(.+)", text, re.IGNORECASE)
    if m:
        fields.installed_location_name = m.group(1).strip().strip("_").strip()

    # --- Model Name -> used for Order_Product__c fuzzy lookup ---
    m = re.search(r"Model\s*Name\s*[:\-]\s*(.+)", text, re.IGNORECASE)
    if m:
        fields.model_name = m.group(1).strip().strip("_").strip()

    # --- Date of Installation -> Installation_Date__c ---
    m = re.search(r"Date\s*of\s*Installation\s*[:\-]?\s*([\d/\-.]{6,10})", text, re.IGNORECASE)
    if m:
        fields.installation_date = _normalize_date_ddmmyy(m.group(1))

    # --- Address -> Installation_Location__c ---
    # Address may wrap onto a second line before the next label (Contact Person).
    m = re.search(
        r"Address\s*[:\-]\s*(.+?)(?=Contact\s*Person|Contact\s*Number|E-?Mail|$)",
        text, re.IGNORECASE | re.DOTALL
    )
    if m:
        addr = " ".join(line.strip() for line in m.group(1).splitlines() if line.strip())
        fields.installation_location = addr.strip()

    # --- Installation Code -> Installation_Code__c ---
    # Hologram box shows two stacked values e.g. "00158" and "C 252" -> "00158-C252"
    m = re.search(r"\b(\d{5})\b.{0,15}\bC\s?(\d{3})\b", text, re.IGNORECASE | re.DOTALL)
    if m:
        fields.installation_code = f"{m.group(1)}-C{m.group(2)}"
    else:
        # fallback: look for two separate short tokens near each other
        code_num = re.search(r"\b(\d{5})\b", text)
        code_c = re.search(r"\bC\s?(\d{3})\b", text)
        if code_num and code_c:
            fields.installation_code = f"{code_num.group(1)}-C{code_c.group(1)}"

    return fields


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/ocr/extract", response_model=OCRResponse)
def extract_document(req: OCRRequest):
    # --- validate ---
    try:
        img_bytes = base64.b64decode(req.image_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    size_mb = len(img_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(status_code=400, detail=f"File exceeds {MAX_FILE_SIZE_MB}MB limit")

    try:
        pil_image = Image.open(io.BytesIO(img_bytes))
        pil_image.verify()
        pil_image = Image.open(io.BytesIO(img_bytes))  # reopen after verify()
    except Exception:
        raise HTTPException(status_code=400, detail="File is not a valid image")

    if pil_image.format not in ALLOWED_FORMATS:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {pil_image.format}")

    # --- preprocess ---
    processed = preprocess_image(pil_image)

    # --- OCR ---
    try:
        if req.mode == "printed":
            raw_text = run_tesseract(processed)
        else:
            raw_text = run_trocr(processed)
    except Exception as e:
        return OCRResponse(success=False, error=f"OCR engine failed: {str(e)}")

    # --- field extraction ---
    fields = extract_fields(raw_text)

    return OCRResponse(success=True, raw_text=raw_text, fields=fields)


@app.get("/health")
def health_check():
    return {"status": "ok"}
