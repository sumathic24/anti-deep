import os
import time
from typing import Any, Dict, Tuple

import cv2
import numpy as np
from PIL import Image

from services.pretrained import VIDEO_MODEL_ID, VideoDetector, load_detector

FRAME_COUNT = 16
MAX_DURATION_SECONDS = 300


class VideoAnalysisService:
    def __init__(self):
        self.detector = load_detector(VideoDetector, VIDEO_MODEL_ID, "VideoService")

    @staticmethod
    def _sample_frames(filepath: str):
        capture = cv2.VideoCapture(filepath)
        if not capture.isOpened():
            raise ValueError("The uploaded file is not a readable video.")
        try:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
            duration = frame_count / fps if fps > 0 else 0
            if duration > MAX_DURATION_SECONDS:
                raise ValueError(f"The video is longer than the {MAX_DURATION_SECONDS}-second limit.")
            indices = np.linspace(0, max(frame_count - 1, 0), FRAME_COUNT, dtype=int)
            frames = []
            for index in np.unique(indices):
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = capture.read()
                if ok:
                    frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            if not frames:
                raise ValueError("The uploaded file contains no readable frames.")
            return frames
        finally:
            capture.release()

    async def analyze_video(self, filepath: str) -> Dict[str, Any]:
        if self.detector is None:
            raise RuntimeError("The video detector is not available. Download the configured checkpoint first.")
        started = time.perf_counter()
        ai_confidence, explainability_regions = self.detector.predict_frames(
            self._sample_frames(filepath)
        )
        return {
            "prediction": "AI-generated video" if ai_confidence > 0.5 else "Real video",
            "ai_confidence": ai_confidence,
            "model_used": self.detector.model_id,
            "gradcam_regions": explainability_regions,
            "explainability": {
                "method": "AEGIS frame-level branch saliency",
                "regions": explainability_regions,
            },
            "processing_time_ms": round((time.perf_counter() - started) * 1000),
        }


def check_video(video_path: str) -> Tuple[str, float]:
    """Classify a video file using the configured detector."""
    if not os.path.isfile(video_path):
        raise ValueError(f"Video file does not exist: {video_path}")
    service = VideoAnalysisService()
    result = service.detector.predict_frames(service._sample_frames(video_path))
    score = float(result[0])
    return ("FAKE" if score > 0.5 else "REAL"), (score if score > 0.5 else 1.0 - score)
