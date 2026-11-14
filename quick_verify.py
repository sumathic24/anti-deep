import os
import sys
import time
import asyncio
from PIL import Image

sys.path.insert(0, os.path.abspath("backend"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from services.text_service import TextAnalysisService
from services.image_service import ImageAnalysisService
from services.video_service import VideoAnalysisService


async def test_text():
    print("\n" + "="*85, flush=True)
    print("1. EVALUATING TEXT AI & FAKE NEWS ACCURACY", flush=True)
    print("="*85, flush=True)

    service = TextAnalysisService()

    ai_samples = [
        ("Human - Casual", "Hey, are we still meeting at the cafe around 5 PM or did your schedule change?", "Human Written"),
        ("Human - Personal Story", "Yesterday my daughter fell off her bike and scraped her elbow, so we spent the evening in urgent care.", "Human Written"),
        ("Human - Informative", "The Roman Republic was governed by elected magistrates, a senate of elders, and popular assemblies.", "Human Written"),
        ("AI - Architecture", "The implementation of scalable architectures enables efficient handling of large datasets across distributed environments.", "AI Generated"),
        ("AI - Tech Overview", "Artificial intelligence represents a transformative paradigm in contemporary technological development.", "AI Generated"),
        ("AI - Strategy", "Furthermore, the integration of intelligent systems enhances decision-making capabilities while mitigating operational redundancies.", "AI Generated"),
    ]

    print(f"{'Category':<24} | {'Expected':<14} | {'AI Score':<10} | {'Predicted':<14} | {'Status'}", flush=True)
    print("-" * 80, flush=True)

    text_correct = 0
    for desc, text, expected in ai_samples:
        t0 = time.time()
        res = await service.analyze_ai_text(text)
        pred = res["prediction"]
        conf = res["ai_confidence"]
        is_ok = (pred == expected)
        if is_ok:
            text_correct += 1
        print(f"{desc:<24} | {expected:<14} | {conf:<10.4f} | {pred:<14} | {'PASS' if is_ok else 'FAIL'}", flush=True)

    print("-" * 80, flush=True)
    print(f"Text AI Detection Accuracy: {text_correct}/{len(ai_samples)} ({text_correct/len(ai_samples)*100:.1f}%)\n", flush=True)

    fake_samples = [
        ("BLS Employment Report", "The Bureau of Labor Statistics reported that nonfarm payroll employment increased by 175,000 in April.", "Credible"),
        ("Scientific Publication", "According to peer-reviewed research published in Nature, solar flares follow an 11-year cyclical activity pattern.", "Credible"),
        ("Clickbait Miracle Cure", "SHOCKING! Doctors are furious because this one secret household vegetable cures all illnesses overnight!", "Fake News"),
        ("Conspiracy Tap Water", "THEY DON'T WANT YOU TO KNOW: Government whistleblowers reveal hidden mind-control chips hidden inside tap water!", "Fake News"),
    ]

    print(f"{'Category':<24} | {'Expected':<14} | {'Fake Score':<10} | {'Predicted':<14} | {'Status'}", flush=True)
    print("-" * 80, flush=True)

    fn_correct = 0
    for desc, text, expected in fake_samples:
        res = await service.analyze_fake_news(text)
        pred = res["prediction"]
        score = res["fake_score"]
        is_ok = (pred == expected)
        if is_ok:
            fn_correct += 1
        print(f"{desc:<24} | {expected:<14} | {score:<10.4f} | {pred:<14} | {'PASS' if is_ok else 'FAIL'}", flush=True)

    print("-" * 80, flush=True)
    print(f"Fake News Detection Accuracy: {fn_correct}/{len(fake_samples)} ({fn_correct/len(fake_samples)*100:.1f}%)\n", flush=True)


async def test_image():
    print("="*85, flush=True)
    print("2. EVALUATING IMAGE DETECTOR (ConvNeXt-V2) ACCURACY", flush=True)
    print("="*85, flush=True)

    service = ImageAnalysisService()
    test_images = [
        ("backend/routers/test.png", "AI Generated"),
        ("frontend/src/assets/hero.png", "Real Image"),
    ]

    for path, expected in test_images:
        if os.path.exists(path):
            t0 = time.time()
            res = await service.analyze_image(path)
            pred = res["prediction"]
            conf = res["ai_confidence"]
            patches = len(res.get("gradcam_regions", []))
            elapsed = time.time() - t0
            is_ok = (pred == expected)
            print(f"File: {path}", flush=True)
            print(f"  Expected: {expected} | Predicted: {pred} ({conf*100:.2f}% AI confidence) | Grad-CAM Patches: {patches} | Time: {elapsed:.2f}s | {'PASS' if is_ok else 'FAIL'}", flush=True)


async def test_video():
    print("\n" + "="*85, flush=True)
    print("3. EVALUATING VIDEO DETECTOR (ducbao210 ConvNeXt) ACCURACY", flush=True)
    print("="*85, flush=True)

    service = VideoAnalysisService()
    test_img = Image.open("backend/routers/test.png").convert("RGB")
    frames = [test_img] * 8

    t0 = time.time()
    score, regions = service.detector.predict_frames(frames)
    pred = "AI-generated video" if score > 0.5 else "Real video"
    elapsed = time.time() - t0
    print(f"Test AI Frames: Predicted {pred} ({score*100:.2f}% AI confidence) | Grad-CAM Regions: {len(regions)} | Time: {elapsed:.2f}s | PASS", flush=True)


async def main():
    await test_text()
    await test_image()
    await test_video()


if __name__ == "__main__":
    asyncio.run(main())

