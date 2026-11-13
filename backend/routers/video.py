import os
import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile

from services.history import history
from services.video_service import VideoAnalysisService

router = APIRouter()
service = VideoAnalysisService()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
MAX_UPLOAD_MB = 200
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


@router.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    extension = os.path.splitext(file.filename or "")[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=422, detail="Unsupported video format. Please upload MP4, MOV, AVI, MKV or WEBM.")
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"The video is larger than {MAX_UPLOAD_MB} MB.")
    filepath = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}{extension}")
    with open(filepath, "wb") as output:
        output.write(data)
    try:
        try:
            result = await service.analyze_video(filepath)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)
    history.record("video", file.filename or "uploaded video", result)
    return result
