from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import io, os, uuid
from services.image_service import ImageAnalysisService
from services.history import history

router = APIRouter()
service = ImageAnalysisService()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Detected file format (not the client-supplied content type) → saved extension
ALLOWED_FORMATS = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp", "GIF": "gif"}


@router.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """Detect if image is AI-generated using EfficientNet + ViT ensemble with GradCAM."""
    data = await file.read()
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.verify()
            fmt = img.format
    except Exception:
        raise HTTPException(status_code=422, detail="The uploaded file is not a valid image.")
    if fmt not in ALLOWED_FORMATS:
        raise HTTPException(status_code=422,
                            detail=f"Unsupported image format ({fmt}). Please upload a JPG, PNG, WEBP or GIF image.")

    filepath = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}.{ALLOWED_FORMATS[fmt]}")
    with open(filepath, "wb") as f:
        f.write(data)

    try:
        result = await service.analyze_image(filepath)
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

    history.record("image", file.filename or "uploaded image", result)
    return result
