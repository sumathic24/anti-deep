from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil, os, uuid
from services.image_service import ImageAnalysisService

router = APIRouter()
service = ImageAnalysisService()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}


@router.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """Detect if image is AI-generated using EfficientNet + ViT ensemble with GradCAM."""
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=422, detail="Unsupported image type.")

    # Save upload
    ext = file.filename.split(".")[-1]
    filename = f"{uuid.uuid4()}.{ext}"
    filepath = os.path.join(UPLOAD_DIR, filename)
    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = await service.analyze_image(filepath)
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

    return result
