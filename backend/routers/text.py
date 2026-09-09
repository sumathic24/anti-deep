from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.text_service import TextAnalysisService

router = APIRouter()
service = TextAnalysisService()


class TextRequest(BaseModel):
    text: str


@router.post("/analyze")
async def analyze_text(req: TextRequest):
    """Detect if text is AI-generated using RoBERTa + LinguaForge-X ensemble."""
    if len(req.text.strip()) < 20:
        raise HTTPException(status_code=422, detail="Text must be at least 20 characters.")
    return await service.analyze_ai_text(req.text)


@router.post("/fake-news")
async def analyze_fake_news(req: TextRequest):
    """Detect misinformation / fake news using BERT-based classifier."""
    if len(req.text.strip()) < 20:
        raise HTTPException(status_code=422, detail="Text must be at least 20 characters.")
    return await service.analyze_fake_news(req.text)
