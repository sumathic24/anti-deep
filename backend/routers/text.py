from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.text_service import TextAnalysisService, ModelUnavailableError
from services.history import history

router = APIRouter()
service = TextAnalysisService()


class TextRequest(BaseModel):
    text: str


@router.post("/analyze")
async def analyze_text(req: TextRequest):
    """Detect if text is AI-generated using RoBERTa + LinguaForge-X ensemble."""
    if len(req.text.strip()) < 20:
        raise HTTPException(status_code=422, detail="Text must be at least 20 characters.")
    try:
        result = await service.analyze_ai_text(req.text)
    except ModelUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    history.record("text", req.text, result)
    return result
