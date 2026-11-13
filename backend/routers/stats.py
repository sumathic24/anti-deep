from fastapi import APIRouter, Query
from services.history import history

router = APIRouter()


@router.get("/stats")
async def get_stats(days: int = Query(7, ge=1, le=90)):
    """Dashboard statistics for the last `days` days, from the recorded analysis history."""
    return history.stats(days)


@router.get("/history")
async def list_history(limit: int = Query(10, ge=1, le=100), offset: int = Query(0, ge=0)):
    """Recorded analyses, newest first."""
    return history.list(limit, offset)


@router.delete("/history")
async def clear_history():
    """Delete every recorded analysis."""
    return {"deleted": history.clear()}
