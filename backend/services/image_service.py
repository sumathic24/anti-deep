import time
from typing import Dict, Any

from services.pretrained import IMAGE_MODEL_ID, ImageDetector, load_detector

class ImageAnalysisService:
    def __init__(self):
        self.detector = load_detector(ImageDetector, IMAGE_MODEL_ID, "ImageService")

    async def analyze_image(self, filepath: str) -> Dict[str, Any]:
        t0 = time.time()
        if self.detector is None:
            raise RuntimeError(f"Image detector {IMAGE_MODEL_ID} is unavailable.")
        ai_score, gradcam_regions = self.detector.predict_with_gradcam(filepath)
        real_score = 1.0 - ai_score
        elapsed_ms = int((time.time() - t0) * 1000)
        return {
            "prediction": "AI Generated" if ai_score > 0.5 else "Real Image",
            "ai_confidence": round(ai_score, 4),
            "real_confidence": round(real_score, 4),
            "model_used": IMAGE_MODEL_ID,
            "vit": {"prediction": "AI Generated" if ai_score > 0.5 else "Real", "confidence": round(ai_score, 4)},
            "gradcam_regions": gradcam_regions,
            "artifacts_detected": ["ViT Grad-CAM attention patches"] if gradcam_regions else [],
            "processing_time_ms": elapsed_ms,
        }