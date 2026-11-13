import os
import time
from typing import Any, Dict

import cv2
import numpy as np
from PIL import Image

from services.pretrained import VIDEO_MODEL_ID, VideoDetector, load_detector

FRAME_COUNT = 16
MAX_DURATION_SECONDS = 300


class VideoAnalysisService:
    def __init__(self):
        self.detector = load_detector(VideoDetector, VIDEO_MODEL_ID, "VideoService")

    def _sample_frames(self, filepath: str):
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
        ai_confidence = self.detector.predict_frames(self._sample_frames(filepath))
        return {
            "prediction": "AI-generated video" if ai_confidence > 0.5 else "Real video",
            "ai_confidence": ai_confidence,
            "model_used": f"{self.detector.model_id} (ConvNeXt frame sampler)",
            "processing_time_ms": round((time.perf_counter() - started) * 1000),
        }
