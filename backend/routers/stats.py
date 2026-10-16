from fastapi import APIRouter
from services.history import history

router = APIRouter()


@router.get("/stats")
async def get_stats():
    """Dashboard statistics computed from the recorded analysis history."""
    return history.stats()
