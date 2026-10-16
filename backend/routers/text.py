from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.text_service import TextAnalysisService, ModelUnavailableError
from services.history import history

router = APIRouter()
service = TextAnalysisService()


class TextRequest(BaseModel):
    text: str
    explain: bool = True  # False skips the SHAP explanation (much faster on large models)


@router.post("/analyze")
async def analyze_text(req: TextRequest):
    """Detect if text is AI-generated, with a SHAP explanation unless explain is false."""
    if len(req.text.strip()) < 20:
        raise HTTPException(status_code=422, detail="Text must be at least 20 characters.")
    try:
        result = await service.analyze_ai_text(req.text, explain=req.explain)
    except ModelUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    history.record("text", req.text, result)
    return result
