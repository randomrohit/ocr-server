# Salesforce OCR Server

Free/open-source OCR backend. Handwriting via TrOCR (Microsoft, HuggingFace),
printed text via Tesseract. OpenCV preprocessing (deskew/denoise/binarize).

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Install Tesseract binary (not a pip package)
# Ubuntu/Debian:
sudo apt install tesseract-ocr
# Mac:
brew install tesseract
# Windows: download installer from
# https://github.com/UB-Mannheim/tesseract/wiki
```

## Run locally

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

First run downloads TrOCR model (~1.3GB) — takes a few minutes.

## Test

```bash
curl -X POST http://localhost:8000/ocr/extract \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "<base64 string>", "mode": "handwritten"}'
```

## Deploy (free tiers)

- **Render.com**: free web service, connect GitHub repo, `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Railway.app**: similar, auto-detects Python
- Note: free tiers sleep on inactivity — first request after idle will be slow (cold start + model load)

## API

`POST /ocr/extract`
```json
{
  "image_base64": "...",
  "file_name": "form1.jpg",
  "mode": "handwritten"   // or "printed"
}
```

Response:
```json
{
  "success": true,
  "raw_text": "...",
  "fields": {
    "name": "John Doe",
    "mobile": "9876543210",
    "address": "123 Main St",
    "amount": "5000"
  }
}
```

## Tuning field extraction

`extract_fields()` in `main.py` uses regex — adjust patterns to match your
actual form layout (labels like "Name:", "Address:" etc.). If your documents
have a fixed layout, consider zone-based cropping before OCR for higher
accuracy instead of running OCR on the whole page.

## Salesforce integration

1. Add Remote Site Setting / Named Credential pointing to this server's URL.
2. Apex `Http` callout sends `image_base64` (compress client-side first,
   watch 6MB callout limit).
3. Parse JSON response, show in LWC verify screen before DML save.
