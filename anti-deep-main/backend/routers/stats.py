from fastapi import APIRouter
import random

router = APIRouter()


@router.get("/stats")
async def get_stats():
    """Dashboard statistics."""
    return {
        "total_analyses": 1247,
        "ai_content_detected": 847,
        "avg_confidence": 91.3,
        "verified_real": 400,
        "text_analyses": 892,
        "image_analyses": 355,
        "weekly": [
            {"day": "Mon", "text": 42, "image": 18},
            {"day": "Tue", "text": 67, "image": 32},
            {"day": "Wed", "text": 55, "image": 25},
            {"day": "Thu", "text": 89, "image": 41},
            {"day": "Fri", "text": 73, "image": 38},
            {"day": "Sat", "text": 94, "image": 52},
            {"day": "Sun", "text": 81, "image": 44},
        ],
        "breakdown": {
            "ai_text": 38,
            "fake_news": 27,
            "ai_image": 21,
            "human_content": 14,
        },
    }
